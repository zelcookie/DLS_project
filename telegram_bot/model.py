from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import misc
import copy


class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target )#to initialize with something

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
        
def gram_matrix(input):
        batch_size , h, w, f_map_num = input.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)

        features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)
    
   
class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
        
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class StyleTransferModel:
    def __init__(self, imsize=128, content_layers=['conv_4'],\
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и тß.д.
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loader = transforms.Compose([
                transforms.Resize(imsize),  # нормируем размер изображения
                transforms.CenterCrop(imsize),
                transforms.ToTensor()])  # превращаем в удобный формат
        
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(self.device)

#     def transfer_style(self, content_img_stream, style_img_stream):
#         # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
#         # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
#         # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
#         # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
#         # В первой итерации, когда вы переносите уже готовую модель из тетрадки с занятия сюда нужно просто
#         # перенести функцию run_style_transfer (не забудьте вынести инициализацию, которая
#         # проводится один раз в конструктор.

#         # Сейчас этот метод просто возвращает не измененную content картинку
#         # Для наглядности мы сначала переводим ее в тензор, а потом обратно
#         return content_img_stream

#     # В run_style_transfer используется много внешних функций, их можно добавить как функции класса
#     # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
#     # принято ставить _ (выглядит это так: def _foo() )
#     # Эта функция тоже не является
    def _get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()]) 
        return optimizer
    
    def _get_style_model_and_losses(self, style_img, content_img):
        cnn = copy.deepcopy(self.cnn)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                #Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        #выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
    def run_style_transfer(self, content_img, style_img, input_img, num_steps=500,
                        style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self._get_style_model_and_losses(style_img, content_img)
        optimizer = self._get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                #взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img
    def process_image(self, img_stream):

        image = Image.open(img_stream)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

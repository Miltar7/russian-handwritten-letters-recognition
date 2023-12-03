from torch import load
import test
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename(initialdir=r"C:\Users\akurt\PycharmProjects\pytorch test")
print(filename)
device='cpu'


with open('model.pt', 'rb') as f:
    
    model=test.Net()
    model.load_state_dict(state_dict=load(f))
    img = Image.open(filename)
    img=img.convert('L')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    letters={0: 'а', 1: 'б', 2: 'в', 3: 'г', 4: 'д', 5: 'е', 6: 'ж', 7: 'з', 8: 'и', 9: 'й', 10: 'к', 11: 'л',
             12: 'м', 13: 'н', 14: 'о', 15: 'п', 16: 'р', 17: 'с', 18: 'т', 19: 'у', 20: 'ф', 21: 'х', 22: 'ц',
             23: 'ч', 24: 'ш', 25: 'щ', 26: 'ъ', 27: 'ы', 28: 'ь', 29: 'э', 30: 'ю', 31: 'я'}
    print(torch.argmax(model(img_tensor)))
    ab=torch.argmax(model(img_tensor))
    num=int(ab.item())
    print(letters[num])

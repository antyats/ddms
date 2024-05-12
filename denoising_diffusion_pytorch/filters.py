from PIL import Image, ImageFilter 
  
def GaussianFilter(img_path):
    image = Image.open(f'{img_path}') 
    smol_image = image.crop((0, 0, 150, 150)) 
    blurred_image = smol_image.filter(ImageFilter.GaussianBlur) 
    image.paste(blurred_image, (0,0)) 

    image.save(f'{img_path}-gaus-filtered.png') 

def SharpenFilter(img_path):
    image = Image.open(f'{img_path}') 
    smol_image = image.crop((0, 0, 150, 150)) 
    sharpen_image = smol_image.filter(ImageFilter.SHARPEN) 
    image.paste(sharpen_image, (0,0)) 

    image.save(f'{img_path}-sharpen-filtered.png') 

def SmoothFilter(img_path):
    image = Image.open(f'{img_path}') 
    smol_image = image.crop((0, 0, 150, 150)) 
    smooth_image = smol_image.filter(ImageFilter.SMOOTH) 
    image.paste(smooth_image, (0,0)) 

    image.save(f'{img_path}-smooth-filtered.png') 

def GrayscaleFilter(img_path):
    image = Image.open(f'{img_path}') 
    smol_image = image.crop((0, 0, 150, 150)) 
    gs_image = smol_image.convert('L')
    image.paste(gs_image, (0,0)) 

    image.save(f'{img_path}-grayscale-filtered.png') 


def SepiaFilter(img_path):
    image = Image.open(img_path)
    width, height = image.size

    pixels = image.load()

    for py in range(height):
        for px in range(width):
            r, g, b = image.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)

    image.save(f'{img_path}-sepia-filtered.png') 

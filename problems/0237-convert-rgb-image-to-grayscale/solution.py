import torch

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or torch.Tensor of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here 
    grayscale = []
    try:
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        for img in image:
            grays = []
            for pixel in img:
                if (pixel[0] < 0 or pixel[0] > 255) or (pixel[1] < 0 or pixel[1] > 255) or (pixel[2] < 0 or pixel[2] > 255):
                    return -1
                gry_pxl = (pixel[0].item() * .299) + (pixel[1].item() * .587) + (pixel[2].item() * .114)
                
                grays.append(round(gry_pxl))

            grayscale.append(grays)

        return grayscale
    except: 
        return -1 

            
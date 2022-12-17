import imageio.v3 as ve 
import PIL.Image as im 

v = ve.imread('random_agent_5.mp4', plugin='pyav')

frame = v[39,:,:,:]
p=im.fromarray(frame)
p.save('env.png')
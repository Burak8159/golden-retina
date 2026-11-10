"""Run one already-cropped eye image through the versioned pupil observer."""
import argparse,json
from PIL import Image
from inference import EyeFinder

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True,help='one eye crop, not a full face/frame');p.add_argument('--eye-side',default='unknown',choices=['left','right','unknown']);p.add_argument('--output');p.add_argument('--device',default='cpu');a=p.parse_args()
    result=EyeFinder(a.checkpoint,a.device).predict(Image.open(a.image),eye_side=a.eye_side)
    text=json.dumps(result,indent=2)
    if a.output:open(a.output,'w').write(text+'\n')
    print(text)
if __name__=='__main__':main()

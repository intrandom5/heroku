import argparse
from STT import STT

def main():
#    parser = argparse.ArgumentParser()
    
#    parser.add_argument('--target', required=True, help='타겟 오디오')
    
#    args = parser.parse_args()
    
#    pred_script = STT(args.target)
    pred_script = STT("0001.wav")
    print(pred_script)
    
if __name__ == "__main__":
    main()

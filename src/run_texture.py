from parse import parse_config
from wonder_tex import WonderTex


if __name__ == '__main__':
    opt = parse_config()

    tex = WonderTex(opt)

    tex.inpaint(opt.prompt, opt.negative_prompt, opt.inference_steps, 
                    no_HD=opt.no_HD, weight_limit=opt.weight_limit)

    
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os 
import random
from datasets import IuxrayTrainDataset,MimicTrainDataset,IuxrayTestDataset,MimicTestDataset
from transformers import AdamW
from transformers import  GPT2Config
from visual_model import CLIPConfig,CLIPEncoder
from tqdm import tqdm
from test_utils import Recorder,top_filtering,compute_scores
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from model import GPT2LMHeadModel
from tokenizer import Tokenizer


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iuxray.json', help='the path to the directory containing the data.')
    #这里要保证数据集中符号和单词之间有空格

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_len', type=int, default=120, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    parser.add_argument('--epochs', type=int, default=20, help='the number of training epochs.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')

    
    parser.add_argument('--lr', type=float, default=3e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ve', type=float, default=5e-6, help='the learning rate for the visual extractor.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')

    
    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--clip_frozen', type=bool, default=False, help='决定图像编码器是否被冻结')

    #test
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--top_k', type=float, default=0, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--no_sample', type=bool, default=True, help='是否取最高概率的预测')
    parser.add_argument('--save_dir', type=str, default='results/best_iuxray/', help='保存模型参数的地址')
    
    parser.add_argument('--temp', default=0.2, type=float)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--gama', default=1, type=float)

    args = parser.parse_known_args()[0]
    return args


#--------------------------------定义一些工具类-------------------------------#
def change_requires_grad(model, req_grad):  #设置模型所有参数是否需要梯度更新
    for p in model.parameters():
        p.requires_grad = req_grad

def save_checkpoint(args, epoch,  model,image_encoder, optimizer,scheduler,best_score,need_update=False):
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'image_encoder': image_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        filename = os.path.join(args.save_dir, 'current_checkpoint.pth')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            print(f"Created directory {args.save_dir}")
        # torch.save(state, filename)
        print("Saving current model...")

        #如果传入的need_update为真，则需要保存当前模型为最好的模型
        if need_update:
            best_path = os.path.join(args.save_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving best model...")

def save_model_params(model, file_path):
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Shape: {param.shape}\n")


def main(args):
    random.seed(args.seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(args.seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(args.seed)   # numpy的随机性
    torch.manual_seed(args.seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device名称:{torch.cuda.get_device_name(0)}')
    print(f'batch_size:{args.batch_size}; lr:{args.lr}; lr_ve:{args.lr_ve}; seed:{args.seed} max_seq_len:{args.max_seq_len};')
    print(f'alpha:{args.alpha}; beta:{args.beta}; gama:{args.gama}; temp:{args.temp} ')
    

    #--------------  定义图像编码器      -------------#
    config = CLIPConfig()
    image_encoder = CLIPEncoder(config,device).img_encoder
    #如果图像编码器冻结
    if args.clip_frozen:  
        change_requires_grad(image_encoder, False)  #将CLIP的参数冻结
        print('冻结BioMedCLIP参数')
    else:
        change_requires_grad(image_encoder, True)
        print('不冻结BioMedCLIP参数')

    #--------------- 定义tokenizer    -------------#
    tokenizer = Tokenizer(args)
    vocab_size = len(tokenizer.idx2token)  #tokenizer的词典大小
    print(f'vocab_size = {vocab_size}')
    special_token_ids = tokenizer.convert_tokens_to_ids(['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>'])
    print(f'special_tokens = {special_token_ids}')
    with open('vocab.txt', 'w', encoding='utf-8') as file:
        for idx, token in tokenizer.idx2token.items():
            file.write(f"{idx}:{token}\n")

    #--------------- 配置gpt2 ----------------#
    config = GPT2Config(vocab_size=tokenizer.get_vocab_size())
    config.add_cross_attention = True
    config.bos_token_id = special_token_ids[1]
    config.eos_token_id = special_token_ids[2]
    model = GPT2LMHeadModel(config)
    model.set_config(args)
    model.resize_token_embeddings(tokenizer.get_vocab_size())
    model = model.to('cuda')
    print("Model Setup Ready...")


    save_model_params(image_encoder, 'image_encoder_params2.txt')
    save_model_params(model, 'gpt2_params.txt')


    #--------------- 加载模型  ------------#
    checkpoint = torch.load(args.save_dir+'model_best.pth')
    model.load_state_dict(checkpoint['model'])
    image_encoder.load_state_dict(checkpoint['image_encoder'])


    #--------------- 打包数据集为batch------------#
    if args.dataset_name == "iu_xray":
       
        test_dataset = IuxrayTestDataset(args, tokenizer, 'test')    #(image, input_ids, report_ids, segment_ids)
        
    else:
        test_dataset = MimicTestDataset(args, tokenizer, 'test')
       
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 1, shuffle=False, 
                                                num_workers=args.num_workers,pin_memory=False) 
   


    
    
    #--------------- 定义测试用的组件------------#
    metrics = compute_scores

    #开始训练
    train_and_val(args,model,image_encoder,test_loader,tokenizer,metrics,device)

def get_report(decoded_sequences):
    if 'so the report is ' in decoded_sequences:
        parts = decoded_sequences.split(" so the report is ")[1:]
        cut_decoded_sequences = " ".join(parts[:]).strip()
    else:
        cut_decoded_sequences = " ".join(decoded_sequences.split()[0:])
    return cut_decoded_sequences

def train_and_val(args,model,image_encoder,test_loader,tokenizer,metrics,device):
    
    print('开始测试')
    model.eval()
    image_encoder.eval()
    
    SPECIAL_TOKENS = ['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    test_gts, test_res = [], [] #分别记录ground truth 和 预测结果

    with open('output/test_output', 'w', encoding='utf-8') as f: 

        for i,batch in enumerate(tqdm(test_loader, desc="Testing")):
            #数据放到device上
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, input_ids, report_ids, segment_ids = batch
            
            # with open('test_dataset.txt','w') as file:
            #     file.write(f'input_ids = {tokenizer.decode(input_ids[0])}\n')
            #     file.write(f'input_ids = {input_ids[0]}\n')
            #     file.write(f'report_ids = {tokenizer.decode(report_ids[0])}\n')
            #     file.write(f'report_ids = {report_ids[0]}\n')
            #     file.write(f'segment_ids = {segment_ids[0]}\n')
            #     print('写入完毕')

            #图像编码
            if args.dataset_name=='iu_xray':
                att_feats_0 = image_encoder(img[:, 0])
                att_feats_1 = image_encoder(img[:, 1])
                img_embeddings = torch.cat((att_feats_0, att_feats_1), dim=1) 
            else:
                img_embeddings = image_encoder(img) 

            current_output = []   #记录生成的序列
            #自回归生成下一步,这里max_seq_len限制了生成内容应该的长度，但是dataset中max_seq_len表示imput+len(生成序列) 
            # max_len = args.max_seq_len - input_ids.shape[1]
            max_len = args.max_seq_len
            with torch.no_grad():
                for step in range(max_len + 1):
                    if step == max_len:  #这里为啥到最后一步就break呢？
                        break
                    #---生成概率并进行一些采样---
                    outputs = model(input_ids=input_ids, 
                                    past_key_values=None, 
                                    attention_mask=None, 
                                    token_type_ids=segment_ids, 
                                    position_ids=None, 
                                    encoder_hidden_states=img_embeddings, 
                                    encoder_attention_mask=None, 
                                    labels=None, 
                                    use_cache=False, 
                                    return_dict=True)
                    lm_logits = outputs.logits 
                    logits = lm_logits[0, -1, :] / args.temperature  #将序列中最后一个词的原始概率分布除以一个温度参数
                    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)  #多样化
                    probs = F.softmax(logits, dim=-1)
                    prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)  #no_sample==true,则取概率最高的，贪婪解码
                    
                    if prev.item() in special_tokens_ids:  
                        break  #如果是特殊标记则不将其加到生成序列后面

                    new_segment = special_tokens_ids[-1]   # <explanation>
                    new_segment = torch.LongTensor([new_segment]).to(device)
                    current_output.append(prev.item())    #新生成的一个单词加到生成序列后面
                    input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)   #新生成的一个单词加到生成序列后面，在下一次循环作为输入
                    segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)

            decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower()
            f.write(f'{decoded_sequences}\n')
            generated_report = get_report(decoded_sequences)
            
            reports = []
            reports.append(generated_report)
            ground_truths = tokenizer.decode_batch(report_ids,skip_special_tokens=True) 
            f.write(f'{reports}\n')
            f.write(f'{ground_truths}\n\n')
            test_res.extend(reports)
            test_gts.extend(ground_truths)

            # break
        
        #----------------一个batch结束-------------------
    
    #每个epoch都计算一下目前的分数,更新log里的记录
    test_scores = metrics({i: [gt] for i, gt in enumerate(test_gts)},{i: [re] for i, re in enumerate(test_res)})
    log = {'test_' + k: v for k, v in test_scores.items()}    
    #输出一下当前log的内容
    for key, value in test_scores.items():
            print('\t{:15s}: {}'.format('test_' + key, value))

       


    


if __name__ == '__main__':

    args = parse_agrs()
    main(args)

    
from transformers import Blip2Processor, Blip2Model,Blip2Config
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
     




    def forward(self, x):
        #x = x.view(x.size(0), -1)  # 将输入展平成一维向量，适应线性层的输入
        x = self.linear1(x)
        #x = self.linear2(x)


        return x




class ModelParallelTemporalQformer(nn.Module):
    def __init__(self):
        super(ModelParallelTemporalQformer,self).__init__()
        device = "cuda" 
        config = Blip2Config()
        test = Blip2Model(config)
        blip2 = Blip2Model.from_pretrained("./blip2-opt-2.7b", torch_dtype=torch.float32)
        blip2.qformer = test.qformer
        self.qformer1 = copy.deepcopy(blip2).to('cuda:0')
        self.qformer2 = copy.deepcopy(blip2).to('cuda:1')
        self.qformer3 = copy.deepcopy(blip2).to('cuda:2')
        self.qformer4 = copy.deepcopy(blip2).to('cuda:2')
        del blip2
     
        self.classifier = SimpleClassifier(4 * 6 * 32 * 768, 2).to(torch.float32)
        pass

    def forward(self,data):
        # [bs, 100, 4, 6, 3, 224, 224]

        query_output = []
        for x in data:
            temporal_size = x.size()[0]
                # GPU split 
            
            res1 = None
            res2 = None
            res3 = None
            res4 = None
            for time in x:
                    v1_input = time[0].to('cuda:0')
                    v2_input = time[1].to('cuda:1')
                    v3_input = time[2].to('cuda:2')
                    v4_input = time[3].to('cuda:3')
                    

                    res1 = self.qformer1(pixel_values = v1_input).last_hidden_state
                    self.qformer2 = self.qformer2.to('cuda:1')
                    res2 = self.qformer2(pixel_values = v2_input).last_hidden_state
                    self.qformer3 = self.qformer3.to('cuda:2')
                    res3 = self.qformer3(pixel_values = v3_input).last_hidden_state
                    self.qformer4 = self.qformer4.to('cuda:3')
                    res4 = self.qformer4(pixel_values = v4_input).last_hidden_state

                    res2 = res2.to("cuda:0")
                    res3 = res3.to('cuda:0')
                    res4 = res4.to('cuda:0')
                    del v1_input,v2_input,v3_input,v4_input


                    
                
                # vehicle_query_output.append(res)
            query_output.append(torch.stack([res1,res2,res3,res4]))
            #print(res)
        query_output = torch.stack(query_output) # [bs, 4 , 6 , 32 , 768]
      
        query_output = query_output.view(query_output.size(0), -1)
        #print(query_output.shape)
        #query_output.unsqueeze(0) # [1,  6 * 32 * 768]
        out = self.classifier(query_output)
        #out.unsqueeze(0)
        return out 

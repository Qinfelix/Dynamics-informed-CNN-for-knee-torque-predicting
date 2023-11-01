import csv  
import numpy as np
import matplotlib.pyplot as plt
import glob2 as glob
import pandas as pd
import torch
from torch import nn
class Footscan_reader():

    def __init__(self):
        self=self
        pass

    #video展示函数
    def show_video(self,video,filepath):
        
        if(video.ndim==3):
            print('开始播放')
            for i in range(video.shape[2]):
                plt.imshow(video[:,:,i])
                plt.savefig(filepath)
            print('完成播放')
        else:
            print('视频矩阵维度错误，无法展示')
        plt.ioff()

    #记录每一帧的函数
    def record_frame(self,video,planar,planar_height,planar_width):
        planar=planar.reshape((planar_height,planar_width,1))
        return np.append(video,planar,axis=2)#拼接到全局变量中去

    #定义读取单个footscan文件的函数，返回一个足压片段和count
    def read_footscan_xls(self,filepath):
        smallflag=0 #每个片段内部每一行记一个，不同片段就归零
        planar=np.array([])#足压图片的暂时存放处
        planar_video = np.array([])#足压视频的存放处
        planar_video_list = [] #足压视频最终存放处
        planar_width =0 
        planar_height =0 
        whichfoot = 2
        footnumber = 0
        footnumber_last = 0
        header_flag = 1
        count=0

        with open(filepath,'r',encoding='gbk') as myFile:  
            lines=csv.reader(myFile)  
            for line in lines: 
                if len(line)!=0 and header_flag==0:#这一行有内容，处于读取状态且不是表头

                    if len(line[0])>30: #此处判断是为了去除frame那一行以及那些说明行，30不会误把窄宽度的足压舍去
                        smallflag+=1
                        line_list = line[0].split('\t')#这一步是把字符串分割为list
                        planar_width=len(line_list)
                        
                        if line_list[0]=='NaN': #该帧无内容
                            smallflag=0

                        else:#为有效帧
                            for i in range(len(line_list)):
                                line_list[i]=float(line_list[i])
                                #这一步是把数字模样的字符串转化为实实在在的数字
                            line_np=np.array(line_list)
                            #这一步是把list转化为ndarray
                            planar=np.append(planar,line_np)
                            #把这一行拼接到图片底部
                        

                    else:#在读取过程中，小于70的一行为说明行，可能是片段切换，也可能是片段内部frame之间的切换，需要分情况处理
                        smallflag=0
                        planar=np.array([]) # 只要出现非数字行就代表一帧图像结束，就要清零
                        if line[0]=='Left foot data' :#代表是片段的切换，以及下一段为左脚
                            footnumber +=1
                            whichfoot = 0
                        elif line[0]=='Right foot data' :#代表是片段的切换，以及下一段为右脚
                            footnumber +=1
                            whichfoot = 1

                        

                elif len(line)==0 :
                #代表出现空行，上一帧截取完毕
                    header_flag=0 #检测到空行就说明表头已经被掠过
                    if(smallflag>0):# 每次都在检测到足压片段末尾的那个空行做操作，其他空行都不管
                        planar_height=smallflag

                        if footnumber==1 and footnumber_last==0: #第一个片段开始
                            planar_video=np.ones((planar_height,planar_width,1))*whichfoot #把每一片段的左右侧信息录入在视频的起始帧
                            planar_video =self.record_frame(planar_video,planar,planar_height,planar_width)#记录第一帧

                        elif footnumber==footnumber_last :#同一个视频的前后帧
                            planar_video =self.record_frame(planar_video,planar,planar_height,planar_width)

                        elif footnumber!=footnumber_last:#片段切换
                            planar_video_list.append(planar_video)#储存上一片段
                            planar_video=np.ones((planar_height,planar_width,1))*whichfoot#生成第二只脚的模板
                            planar_video =self.record_frame(planar_video,planar,planar_height,planar_width)#记录第二只脚的第一帧
                
                        else :
                            print('读取错误')

                        footnumber_last=footnumber #每检测到一个足压片段更新footnumber一次

                if header_flag ==1 : #表示是表头，我要用表头提取count信息
                    if line[0][0]=='D':#代表是第一行
                        for i in range(len(line[0])):
                            if line[0][i].isdigit()==True:
                                if line[0][i+1].isdigit()==True:
                                    count=int(line[0][i:i+2])
                                else:
                                    count=int(line[0][i])
                                break

            planar_video_list.append(planar_video)#储存最后一个片段

        return planar_video_list,count

    #对单个footscan同时读取左右脚的函数，返回值为一个list——【左脚video，右脚video】,加上这个文件的count
    def read_bothside(self,filepath):
        #但注意排列组合要按照左右脚的顺序，而不能按照患侧脚和非患侧脚的顺序，同时把week_list要赋予左右脚的信息因子做以标志
        footvideo,count=self.read_footscan_xls(filepath)
        left_list=[]
        right_list=[]
        return_list=[]
        for video in footvideo:
            if video.ndim==3 and video.shape[0]>29 and video.shape[1]>14: #除去无效的片段
                    if video[1,1,0]==0:
                        left_list.append(video)
                    else:
                        right_list.append(video)
        
        if len(left_list)*len(right_list)>0:#代表均有有效片段，下面就进行排列组合
            for i in range(len(left_list)):
                for j in range(len(right_list)):
                    return_list.append((left_list[i],right_list[j]))
        else:
            print('该测试趟中足压片段只检测到了一只脚,导致读取的左右脚组合片段个数为0')
        return return_list,count


    #对视频在height和width维度进行填充的函数
    def video_central_padding(self,input,target_height,target_width):
        if input.ndim!=3:
            print('zeropading输入的矩阵维度错误')
        else:
            (height,width,timestep)=np.shape(input)
            pad_h=target_height-height
            pad_w=target_width-width
            pad_left=int(pad_w/2)
            pad_right=pad_w-pad_left
            pad_top=int(pad_h/2)
            pad_bottom=pad_h-pad_top
            padder =nn.ZeroPad2d((pad_left,pad_right,pad_top,pad_bottom))
            output = torch.zeros((timestep-1,target_height,target_width))
            for i in range(timestep-1):
                output[i,:,:]= padder(torch.tensor(input[:,:,i+1]))
            
            return output
    
    '''
    在填充这一步给数据做数据增强,可通过左中右的平移、和上下、左右、不翻折的翻折,得到九种组合，进而将数据量增加九倍
    '''
    def video_augment_padding(self,input,target_height,target_width):
        if input.ndim!=3:
            print('zeropading输入的矩阵维度错误')
        else:
            (height,width,timestep)=np.shape(input)
            video_list=[]
            for i in [2,4,4/3]:
                pad_h=target_height-height
                pad_w=target_width-width
                #此时就会有左中右三种方式
                pad_left=int(pad_w/i)
                pad_right=pad_w-pad_left
                pad_top=int(pad_h/2)
                pad_bottom=pad_h-pad_top
                padder =nn.ZeroPad2d((pad_left,pad_right,pad_top,pad_bottom))
                output1 = torch.zeros((timestep-1,target_height,target_width))
                output2 = torch.zeros((timestep-1,target_height,target_width))
                output3 = torch.zeros((timestep-1,target_height,target_width))
                #此处会有原图，上下翻折，左右翻折三种方式
                for i in range(timestep-1):
                    output1[i,:,:]= padder(torch.tensor(input[:,:,i+1]))
                    output2[i,:,:]=torch.flip(output1[i,:,:],dims=[0])
                    output3[i,:,:]=torch.flip(output1[i,:,:],dims=[1])
                video_list.extend([output1,output2,output3])

            return video_list
    
    # 裁剪视频（只留取中间部分）
    def cut_video(self,video,norm_length):
        video_length=video.shape[0]
        if video.ndim!=3 or video_length < norm_length:
            raise TypeError('cut video is not in correct form')
        height=video.shape[1]
        width=video.shape[2]
        begin_index=video_length-norm_length
        begin_index = begin_index // 2
        after_cut=torch.zeros((norm_length,height,width))
        for i in range(norm_length):
            after_cut[i]=video[i+begin_index]
        return after_cut
    
    #对视频时间轴上做normalization，在步态周期平均抽norm_length个点数，线性插值得到norm之后的数据
    def normalize_video(self,video,norm_length):
        period=video.shape[0]-1
        if video.ndim!=3 :
            raise TypeError('cut video is not in correct form')
        #我都默认原始的采样频率是1，依次建立时间轴
        delta=period/(norm_length-1)
        height=video.shape[1]
        width=video.shape[2]
        ans=torch.zeros((norm_length,height,width))
        for i in range(norm_length):
            t=i*delta
            left_point=t//1
            left_distance=t-left_point
            right_distance=1-left_distance
            if int(left_point)==period:
                ans[i,:,:]=video[int(left_point),:,:]
            else:
                ans[i,:,:]=video[int(left_point),:,:]*right_distance + video[int(left_point)+1,:,:]*left_distance
        return ans,delta



    '''
    获得某一个具体的data的接口
    para—— task:"walk/run",catrgory:"A/B/C/ABC/vicon",name:'患者的名字/ALL',week:'12/24/36/48/ALL'
    return——train_x,train_y均为tensor,维度分别为(batch,timestep,channel=1,height,width)和(batch)
    路径定义在函数内部，不作为参数传入
    '''
    def get_data(self,task,category,name,week,augment):
        if week=='ALL':
            all_weeks=['12','24','36','48']
        else:
            all_weeks=[week]
        input_list = []
        week_list=[]
        task = task
        file_number=0
        count_list=[]

        #开始读取
        print('开始寻找  {}  在  {}  week康复周次的足压数据,具体的task为{},分组是{}'.format(name,week,task,category))
        for week in all_weeks:
            if name=='ALL':
                footscan_path_l = glob.glob(r'D:\\data\\DJO\\足压\\{}\\L\\*\\{}\\{}\\*.xls'.format(category,week,task))
                footscan_path_r = glob.glob(r'D:\\data\\DJO\\足压\\{}\\R\\*\\{}\\{}\\*.xls'.format(category,week,task))
            else:
                footscan_path_l = glob.glob(r'D:\\data\\DJO\\足压\\{}\\L\\{}\\{}\\{}\\*.xls'.format(category,name,week,task))
                footscan_path_r = glob.glob(r'D:\\data\\DJO\\足压\\{}\\R\\{}\\{}\\{}\\*.xls'.format(category,name,week,task))
            file_number += len(footscan_path_l) + len(footscan_path_r) 

            #左脚伤为week为-12/-24/-36/-48 ，右脚伤为正数，总共有八类，此时的video combine是个list，第一项是左脚，第二项右脚，且没有经过任何预处理
            for path in footscan_path_l:
                video_combine,count=self.read_bothside(path)
                if len(video_combine)!=0:
                    input_list.extend(video_combine)
                    week_list.extend([int(week)*(-1)]*len(video_combine))
                    count_list.extend([count]*len(video_combine))


            for path in footscan_path_r:
                video_combine,count=self.read_bothside(path)
                if len(video_combine)!=0:
                    input_list.extend(video_combine)
                    week_list.extend([int(week)*1]*len(video_combine))  
                    count_list.extend([count]*len(video_combine))  

        if file_number == 0:
            print('该患者在该时间没有测量足压')
            print('\n')
            return 0,0,0,0
            
            
        else:
            #读取的统计信息输出
            print('该患者在该时间点存在记录的足压数据')
            # print('信息读取完毕，下展示读取数据的统计信息')
            # counter = pd.value_counts(week_list)
            # print('各标签数据分布：',counter)
            print('此次测量患者总共测量了 {} 趟，平均每一趟的等效步态周期个数是{}个'.format(file_number,len(week_list)/file_number))

            #数据筛选
            print('进行残缺数据筛选')
            good_input=[]
            good_week=[]
            good_count=[]
            bad_input=[]
            max_timestep=120
            min_timestep=60
            max_width=35
            min_width=15
            max_height=50
            min_height=30

            norm_length = 101
            
            # print('筛选前的左右脚组合足压片段有{}个'.format(len(input_list)))
            for i in range(len(input_list)) :
                (left_height,left_width,left_timestep) = np.shape(input_list[i][0])
                (right_height,right_width,right_timestep) = np.shape(input_list[i][1])

                if max(left_timestep,right_timestep) < max_timestep and min(right_timestep,left_timestep)>min_timestep :
                    if max(left_width,right_width) < max_width and min(right_width,left_width)>min_width :
                        if max(left_height,right_height) < max_height and min(right_height,left_height)>min_height :
                            good_input.append(input_list[i])
                            good_week.append(week_list[i])
                            good_count.append(count_list[i])
                        else:
                            bad_input.append(input_list[i])
                    else:
                        bad_input.append(input_list[i])
                else :
                    bad_input.append(input_list[i])
            print('其中有  {}  组数据中的某一片段有残缺'.format(len(bad_input)))
            # print('筛选后可以用的片段数量有{}个'.format(len(good_input)))
            delta_list=[]
            
            #在height和width维度做zeropadding 并且裁剪出来有效的片段
            print('进行数据的填充与拼接')
            input_tensor_list=[]
            if augment == 0:
                print('过程中无数据增强')
            else:
                print('过程中进行数据增强')
            for combine_video in good_input:
                if augment == 0:
                    #在padding之前，足压片段的维度都是（height，width，timestep),padding之后就变成了（timestep，height，width)便于之后训练
                    left_video=self.video_central_padding(combine_video[0],max_height,max_width)
                    right_video=self.video_central_padding(combine_video[1],max_height,max_width)
                    left_video,left_delta=self.normalize_video(left_video,norm_length)
                    right_video,right_delta=self.normalize_video(right_video,norm_length)
                    tensor = torch.cat((left_video,right_video),0)
                    input_tensor_list.append(tensor)
                    delta_list.append((left_delta,right_delta))
                else:
                    left_augment_list=self.video_augment_padding(combine_video[0],max_height,max_width)
                    right_augment_list=self.video_augment_padding(combine_video[1],max_height,max_width)
                    for i in range(len(left_augment_list)):
                        left_video,left_delta=self.normalize_video(left_augment_list[i],norm_length)
                        right_video,right_delta=self.normalize_video(right_augment_list[i],norm_length)
                        tensor = torch.cat((left_video,right_video),0)
                        input_tensor_list.append(tensor)
                        delta_list.append((left_delta,right_delta))
            
            train_x=torch.stack(input_tensor_list) 
            train_x=torch.unsqueeze(train_x,2)
            print('最终完成填充与拼接,最终得到的input data的维度为',train_x.shape)

            #处理week和count
            # for i in range(len(good_week)):
            #     if good_week[i]<0:
            #         good_week[i]=good_week[i] /12+4
            #     else:
            #         good_week[i]=good_week[i] /12+3
            
            if augment==1:
                week=np.array(good_week)
                week=np.expand_dims(week,1).repeat(9,axis=1)
                week=week.reshape(-1,1)
                week=torch.tensor(week,dtype=torch.long)   

                count=np.array(good_count)
                count=np.expand_dims(count,1).repeat(9,axis=1)
                count=count.reshape(-1,1)
                count=torch.tensor(count,dtype=torch.long)  
            else:
                week=torch.tensor(good_week,dtype=torch.long)  
                count=torch.tensor(good_count,dtype=torch.long)  


            return train_x,week,count,delta_list
        



class GAIT_DATA():
    def __init__(self,category,name,week,count,data):
        self=self
        self.category=category
        self.name=name
        self.week=week
        self.count=count
        self.data=data

class gait_data_reader():
    def __init__(self):
        self=self
        pass


    '''
    统计某一读取批次的步态数据，计算此批数据共包含了多少次回访，并返回这些回访信息，便于匹配相应的足压数据
    '''
    def get_name_week_list(self,gait_data_list,info_dic):
        name_week_list=[]
        name_list=[]
        category_list=[]
        for gaitdata in gait_data_list:
            name_week=(info_dic.get(gaitdata.name),str(gaitdata.week*12))
            category=gaitdata.category
            #判断名字是否出现过
            if name_week[0] in name_list:
                pass
            else:
                name_list.append(name_week[0])
            #判断此次回访是否出现过
            if name_week in name_week_list:
                pass
            else:
                name_week_list.append(name_week)
            if len(category)!=6:
                print(name_week,gaitdata.count,category)


        print('该部分数据中共有 {} 位患者，平均每位患者有{}次回访，共计{}次,平均每次回访测试了{}趟，共计{}趟'.format(len(name_list),len(name_week_list)/len(name_list),len(name_week_list),len(gait_data_list)/len(name_week_list),len(gait_data_list)))
        print('\n \n')
        return name_week_list


    '''
    读取单个topic的gaitdata文件的函数,输入是文件路径,输出是一个list,list的每一个量都是特定一次回访的特定一趟的该主题下的所有种类的数据（全部集成在一个gaitdata结构体中，category是一个list，data的列数目就是模拟量种类数目*3）
    '''
    def read_file(self,filepath):
        file=open(filepath,encoding='gbk')
        #先读出所有数据
        all_rows=[]
        for line in file.readlines():
            row=line.split('\t')
            all_rows.append(row)
        row_length=len(all_rows[0])
        row_num=len(all_rows)

        #对前5行进行操作，提取出D1W04+T1+W1+data_name(ex.pelvsi cog)这四个信息，每一项都是一个tuple（name，week，count,category）
        info_row=all_rows[0]
        info_list=[]
        category_list=[]
        file_begin_index=[1]
        last_file_info=info_row[1]
        for i in range(1,row_length,3):

            this_info=info_row[i]
            #category一定在第二行
            category=all_rows[1][i]

            
            #记录每一个文件的数据的开始(上一个步态文件的全部数据均读取完毕)
            if this_info!=last_file_info:
                file_begin_index.append(i)
                #name的提取没有什么大问题_基本都在第一行的前五个
                name=last_file_info[0:5]
                #week不一定都在一个固定的位置，但是一定是name后面的第一个数字
                for j in range(5,15):
                    if last_file_info[j].isdigit()==True:
                        week=int(last_file_info[j:j+1])
                        break
                #count一定在最后的.c3d前面一个，注意区分两位数和一位数就好
                if last_file_info[-6]=='1':
                    count=int(last_file_info[-6:-4])
                else:
                    count=int(last_file_info[-5])
                info_list.append((name,week,count,category_list))
                category_list=[]

            #最后一组
            if i==row_length-3:
                category_list.append(category)
                #name的提取没有什么大问题_基本都在第一行的前五个
                name=last_file_info[0:5]
                #week不一定都在一个固定的位置，但是一定是name后面的第一个数字
                for j in range(5,15):
                    if last_file_info[j].isdigit()==True:
                        week=int(last_file_info[j:j+1])
                        break
                #count一定在最后的.c3d前面一个，注意区分两位数和一位数就好
                if last_file_info[-6]=='1':
                    count=int(last_file_info[-6:-4])
                else:
                    count=int(last_file_info[-5])
                info_list.append((name,week,count,category_list))
                category_list=[]


            last_file_info=this_info

            category_list.append(category)
               
        print('在文件路径{}下，共读取了{}个步态文件'.format(filepath,len(info_list)))


        #对后面的数据行进行处理，集成到一个ndarrary当中，大小帧数*3（3代表xyz三个方向）
        data_list=[]

        for i in range(len(file_begin_index)):
            #读取一个文件的所有data
            begin_index=file_begin_index[i]
            if i==len(file_begin_index)-1:
                end_index=row_length
            else:
                end_index=file_begin_index[i+1]
            
            column_num=end_index-begin_index
            buffer=[]
            for row_index in range(5,row_num):
                #先检测后期出现无数据行
                if row_index>1500 and all_rows[row_index][begin_index:end_index]==['']*column_num:
                    break
                else:
                    buffer.extend(all_rows[row_index][begin_index:end_index])
            #下面将读取的数据都转化为浮点数，如果遇到数据缺失，就视为和上一帧数据一样
            float_buffer=[]
            for i in range(len(buffer)):
                if buffer[i]=='':
                    buffer[i]=0
                try:
                    float(buffer[i])
                    float_buffer.append(float(buffer[i]))
                except:
                    try:
                        a=float_buffer[i-column_num]
                        float_buffer.append(a)
                    except:
                        float_buffer.append(0)
                        print('{}文件中出现错误，此时的gaitdata 表格中出现空格数据，而且前面没有可参考值'.format(filepath))
            data=np.array(float_buffer,ndmin=1)
            data=data.reshape(-1,column_num)
            data_list.append(data)


        
        #把上面的两个集成在一个结构体内
        gait_data_list=[]
        for i in range(len(info_list)):
            if len(info_list[i][3])==6:
                gaitdata=GAIT_DATA(info_list[i][3],info_list[i][0],info_list[i][1],info_list[i][2],data_list[i])
                gait_data_list.append(gaitdata)
            else:
                print('在{}中出现残缺文件，被删除'.format(filepath))

        return gait_data_list
    
    '''
    该函数就是对目前已有的所有数据进行读取,返回一个list,list的元素是GAITDATA结构体,每一个元素是特定一次回访（name，week、count唯一确定）与该主题（all_angle,grf_exp）相关的所有模拟量的集合——此时的category是一个list，里面集合了包含的所有模拟量的种类；data是一个【m,3*n】m是帧数，n是该主题下的模拟量种类
    此时路径定义在函数内部,并不作为参数传入

    '''
    def get_data(self,data_content):
        print('开始读取{}数据'.format(data_content))
        gaitdata_filepath=glob.glob(r'D:\\data\\dynamic\\gait\\*\\*\\{}.txt'.format(data_content))
        gaitdata_list=[]
        for filepath in gaitdata_filepath:
            gaitdata=self.read_file(filepath)
            gaitdata_list.extend(gaitdata)

        
        print('读取完毕，共读取{}个步态文件中的{}数据'.format(len(gaitdata_list),data_content))
        print('\n','*'*50,'\n')
        return gaitdata_list
    

    def read_motsto(self,filepath):
        file=open(filepath,encoding='gbk')
        #先读出所有数据
        all_rows=[]
        for line in file.readlines():
            row=line.split('\t')
            all_rows.append(row)
        row_num=len(all_rows)
        enheader_index=0
        #先获得enheader在哪一行
        for i in range(row_num):
            if all_rows[i][0]=='endheader\n':
                enheader_index=i
                break
        category_list=all_rows[enheader_index+1]
        column_num=len(category_list)



        buffer=[]
        for row_index in range(enheader_index+2,row_num):
            #先检测后期出现无数据行
            if row_index>100 and all_rows[row_index][:]==['']*column_num:
                break
            else:
                buffer.extend(all_rows[row_index][:])

        #下面将读取的数据都转化为浮点数，如果遇到数据缺失，就视为和上一帧数据一样
        float_buffer=[]
        for i in range(len(buffer)):
            if buffer[i]=='':
                buffer[i]=0
            try:
                float(buffer[i])
                float_buffer.append(float(buffer[i]))
            except:
                try:
                    a=float_buffer[i-column_num]
                    float_buffer.append(a)
                except:
                    float_buffer.append(0)
                    print('{}文件中出现错误，此时的gaitdata 表格中出现空格数据，而且前面没有可参考值'.format(filepath))
        data=np.array(float_buffer,ndmin=1)
        data=data.reshape(-1,column_num)
        
        gaitdata=GAIT_DATA(category_list,'unknown','unknown','unkown',data)
        return gaitdata
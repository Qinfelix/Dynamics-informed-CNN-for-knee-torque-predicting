import math
import matplotlib.pyplot as plt
import numpy as np


class COMPARATOR():

    def __init__(self):
        self=self
        
        category_dic={}
        category_dic['knee_angle_r_moment']=15
        category_dic['knee_angle_l_moment']=6
        category_dic['knee_flexion_r_moment']=15
        category_dic['knee_flexion_l_moment']=6
        category_dic['knee_adduction_r_moment']=16
        category_dic['knee_adduction_l_moment']=7
        category_dic['ankle_angle_r_moment']=9
        category_dic['ankle_angle_l_moment']=0
        category_dic['subtalar_angle_r_moment']=11
        category_dic['subtalar_angle_l_moment']=2
        category_dic['hip_flexion_r_moment']=12
        category_dic['hip_flexion_l_moment']=3
        category_dic['hip_adduction_r_moment']=13
        category_dic['hip_adduction_l_moment']=4


        
        self.category_dic=category_dic

    def find_this_category(self,gaitdata,data_category):
        category_list=gaitdata.category
        for i in range(len(category_list)):
            if category_list[i]==data_category:
                return i
        
    def get_nrmse_rmse_cor(self,computed_gaitdata,reference_gaitdata,BW,Ht,data_category,plotornot,idorra='id'):
        computed_alldata=computed_gaitdata.data

        #对reference做单位变换
        original_ref=reference_gaitdata.data
        original_ref_Nm=original_ref*BW*Ht

        #抽取得到对应数据类型的那一列
        
        ori_computed_result=computed_alldata[:,self.find_this_category(computed_gaitdata,data_category)]
        ori_ref_result=original_ref_Nm[:,self.category_dic[data_category]]


        #对于ori_ref_result后面会有一段全是0的，可以忽略
        invalid_flag=0
        for i in range(10):
            if int(ori_ref_result[-(i+1)])==0:
                invalid_flag+=1
        
        if invalid_flag==10:#代表最后0.1s全是0，可以判断为无效片段，进行裁剪：
            invalid_length=10
            for l in range(11,len(ori_ref_result)):
                if int(ori_ref_result[-l])==0:
                    pass
                else:
                    invalid_length=l-1
                    break
            ori_ref_result=ori_ref_result[0:len(ori_ref_result)-invalid_length]

        
        #ref的频率是1000Hz，将其降低为100Hz
        ref_result=ori_ref_result[0:len(ori_ref_result):10]
        time=np.arange(len(ref_result))*0.01

        #对rra进行抽样
        if idorra=='rra':
            sampled_result=[]
            for j in range(len(ref_result)):
                ref_time=j*0.01
                for i in range(len(ori_computed_result)):
                    this_time=computed_alldata[i,0]
                    consistency=this_time-ref_time
                    if consistency>=0:
                        sampled_result.append(ori_computed_result[i])
                        break
            if len(sampled_result)!=len(ref_result):
                sampled_result.append(0)
            computed_result=np.array(sampled_result)
        else:
            if len(ref_result)<=len(ori_computed_result):
                computed_result=ori_computed_result[0:len(ref_result)]
            else:#表示此时computed的反而比ref要少
                computed_result=ori_computed_result
                time=time[0:len(computed_result)]
                ref_result=ref_result[0:len(computed_result)]

        
        
        #对某些列表要做方向对折
        toggle_category=['knee_angle_r_moment','knee_angle_l_moment','knee_flexion_r_moment','knee_flexion_l_moment','knee_adduction_l_moment'
                        ,'subtalar_angle_r_moment','subtalar_angle_l_moment']
        if data_category in toggle_category:
            computed_result=-computed_result


        if computed_result.shape[0]!=ref_result.shape[0]:
            return '两者数据长度不同,reference变换后为{}, computed则为{}'.format(ref_result.shape,computed_result.shape)
        else:
            if plotornot==1:
                plt.plot(time,computed_result,label='computed')
                plt.plot(time,ref_result,label='reference')
                plt.xlabel('time(s)')
                plt.ylabel(data_category)
                plt.legend()
                plt.show()
            
            corr=np.corrcoef(computed_result,ref_result)
            corr_value=corr[0,1]
            if corr_value<0:
                computed_result=-computed_result
            mse = sum([(x - y) ** 2 for x, y in zip(computed_result, ref_result)]) / len(computed_result)
            rmse=math.sqrt(mse)
            nrmse=rmse/(max(ref_result)-min(ref_result))
            return nrmse,math.sqrt(mse),corr_value

            
class CONVERTOR():
    def __init__(self):
        self=self

    def grf2sto(self,grf_gaitdata,sample_frequency):
        file_name=grf_gaitdata.name+'T'+str(grf_gaitdata.week)+'W'+str(grf_gaitdata.count)+'.sto'
        file_path='D:\\data\\dynamic\\grffile\\'+file_name
        
        data=grf_gaitdata.data
        category=grf_gaitdata.category
        nRow=data.shape[0]
        nColumns=data.shape[1]
        nCategories=len(category)
        grf_file=open(file_path,'w')
        grf_file.write('ground_reaction\n')
        grf_file.write('version=1\n')
        grf_file.write('nRows='+str(nRow)+'\n')
        grf_file.write('nColumns='+str(nColumns+1)+'\n')
        grf_file.write('inDegrees=no\n')
        grf_file.write('endheader\n')

        enheader='time\t ground_force_l_vx \t ground_force_l_vy \t ground_force_l_vz \t'
        enheader+= 'ground_force_l_px \t ground_force_l_py \t ground_force_l_pz \t'
        enheader+='ground_torque_l_x \t ground_torque_l_y \t ground_torque_l_z \t'
        enheader+='ground_force_r_vx \t ground_force_r_vy \t ground_force_r_vz \t'
        enheader+= 'ground_force_r_px \t ground_force_r_py \t ground_force_r_pz \t'
        enheader+='ground_torque_r_x \t ground_torque_r_y \t ground_torque_r_z \n'
        grf_file.write(enheader)
        change_category_data=np.zeros((nRow,nColumns))
        new_data=np.zeros((nRow,nColumns))
        change_category_seq=[0,2,4,1,3,5]
        #做category的变换
        for i in range(6):
            ori_index=change_category_seq[i]
            change_category_data[:,3*i:3*i+3]=data[:,ori_index*3:ori_index*3+3]
        

        #坐标变换
        for i in range(nCategories):
            #和trc一致，绕着x轴旋转-90度
            new_data[:,3*i]=change_category_data[:,3*i]
            new_data[:,3*i+1]=change_category_data[:,3*i+2]
            new_data[:,3*i+2]= - change_category_data[:,3*i+1]

        
        delta=1/sample_frequency
        for time_index in range(nRow):
            toWrite = str(delta*(time_index))
            for angle_index in range(nColumns):
                toWrite += '\t' + str(new_data[time_index,angle_index])
            grf_file.write(toWrite + '\n')
        
        grf_file.close()
        print(file_name+' writing is finished')

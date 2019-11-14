import pandas as pd
import os
from pytools import seg as pg


class ChangeScore(object):
    def __init__(self):
        self.__segFields = []
        self._generate_origi_score_list_way='near'
        self._input_dataframe=None
        self._input_origi_score_list_dataframe=None
        self._output_stand_score_list_dataframe=None
        self._segOrigiMax={'yw':150,'sx':150,'wy':150,'wl':110,'sw':90,'hx':100,'dl':100,'ls':100,'zz':100}
        self._segOrigiMin={'yw':0,'sx':0,'wy':0,'wl':0,'sw':0,'hx':0,'dl':0,'ls':0,'zz':0}
        self._segObjMax=100
        self._segObjMin=20
        self._segNumber=8
        self._percentList=[0.03, 0.07, 0.16, 0.24, 0.24, 0.16, 0.07, 0.03]
        self._segStandList=[20, 30, 40, 50, 60, 70, 80, 90, 100]
        self._seg_origi_standlist={} # 原始分数点列表
        self.__output_segstandList_dataframe = None  # 分数
        self._score_dict={}
    @property
    def output_data(self):
        return self._output_stand_score_list_dataframe

    def set_parameters(self,segObjMax=None,segObjMin=None,segNumber=None,percentList=None,segStandList=None,generate_origi_score_list_way=None):
        if isinstance(segObjMax, int):
            self._segObjMax=segObjMax
        if isinstance(segObjMin, int):
            self._segObjMin=segObjMin
        if isinstance(segNumber, int):
            self._segNumber=segNumber
        self._generate_origi_score_list_way=generate_origi_score_list_way
    # 设置一分一段表的DF
    def set_data_yfyd(self,df, field_list=None):
        self._input_dataframe = df
        self.__segFields= field_list

    # 设置要转换的原始分数的DF
    def set__input_origi_score_list_dataframe(self,df):
        self._input_origi_score_list_dataframe = df

    # 设置按比例求出的分数点
    def set_score_dict(self):
        for f in self.__segFields:
            temp_origi_standlist=self._seg_origi_standlist.get(f)
            # print(temp_origi_standlist)
            temp_dict={}
            for i in range(len(temp_origi_standlist)):
                temp_dict[temp_origi_standlist[i]]=self._segStandList[i]
            self._score_dict[f]=temp_dict
    # 四舍五入
    def my_round(self,number, point):
        temp = 10 ** point
        temp_int = int(number * temp)
        diff = abs(number * temp - temp_int)
        if number >= 0:
            result = temp_int / temp if diff < 0.5 else (temp_int + 1) / temp
        else:
            result = temp_int / temp if diff < 0.5 else (temp_int - 1) / temp
        return result
    # 转换分数
    def getEveryScore(self,x,f,which_point):
        temp_dict=self._score_dict.get(f)
        temp_seg_origi_standlist=self._seg_origi_standlist.get(f)
        if x in temp_dict:
            return temp_dict.get(x)  # 如果分数正好在分段点上，直接返回对应分数
        previous=0
        next=0
        not_found=True
        for i in range(len(temp_seg_origi_standlist)):
            if not_found:
                next=temp_seg_origi_standlist[i]
                if x<next:
                    not_found=False
                    previous=temp_seg_origi_standlist[i-1]
            else:
                break
        previous_value=self._score_dict.get(f).get(previous)
        next_value=self._score_dict.get(f).get(next)

        return self.my_round((x-previous)*(next_value-previous_value)/(next-previous)+previous_value,which_point)

    # 转换分数
    def generate_stand_score(self,which_point):
        self.set_score_dict()
        self._output_stand_score_list_dataframe = self._input_origi_score_list_dataframe.copy()
        for f in self.__segFields:
            self._output_stand_score_list_dataframe[f+'_stand']=self._output_stand_score_list_dataframe[f].apply(self.getEveryScore,args=(f,which_point))
        print(self._output_stand_score_list_dataframe)

    # def set_origi_standList(self,stand_list):
    #     self._seg_origi_standlist = stand_list

    # 根据比例划分找出对应的原始分数
    def generate_OrigiScoreList(self):
        cussum = pd.Series(self._percentList).cumsum()
        if self._generate_origi_score_list_way.lower()=='near':
            for f in self.__segFields:
                tempresult=[]
                # tempresult.append(self._segOrigiMin.get(f))# 最小的放进来  段点是0-150 或者0-100的情况
                tempresult.append(self._input_origi_score_list_dataframe[f].min())

                for i in range(len(cussum)-1):
                    tempLess=self._input_dataframe[self._input_dataframe[f+'_percent']<=cussum[i]]
                    tempLessPoint=tempLess[(tempLess[f+'_percent']==tempLess[f+'_percent'].max()) & (tempLess[f+'_count']!=0)]

                    tempMore=self._input_dataframe[self._input_dataframe[f+'_percent']>cussum[i]]
                    tempMorePoint=tempMore[(tempMore[f+'_percent']==tempMore[f+'_percent'].min()) &  (tempMore[f+'_count']!=0)]

                    if ( tempMorePoint[f+'_percent'].values[0]-cussum[i] )<(cussum[i]- tempLessPoint[f+'_percent'].values[0]) :
                        tempresult.append(tempMorePoint['seg'].values[0])
                    if ( tempMorePoint[f+'_percent'].values[0]-cussum[i] )> (cussum[i]- tempLessPoint[f+'_percent'].values[0]) :
                        tempresult.append(tempLessPoint['seg'].values[0])
                    if ( tempMorePoint[f+'_percent'].values[0]-cussum[i] )== (cussum[i]- tempLessPoint[f+'_percent'].values[0]) :
                        tempresult.append(tempMorePoint['seg'].values[0])
                tempresult.append(self._input_origi_score_list_dataframe[f].max())
                print(tempresult)
                self._seg_origi_standlist[f] =tempresult

        if self._generate_origi_score_list_way.lower() == 'the_biggest_smaller_than_or_equal':
            for f in self.__segFields:
                tempresult = []
                # tempresult.append(self._segOrigiMin.get(f))# 最小的放进来  段点是0-150 或者0-100的情况
                tempresult.append(self._input_origi_score_list_dataframe[f].min())

                for i in range(len(cussum) - 1):
                    tempLess = self._input_dataframe[self._input_dataframe[f + '_percent'] <= cussum[i]]
                    tempLessPoint = tempLess[
                        (tempLess[f + '_percent'] == tempLess[f + '_percent'].max()) & (tempLess[f + '_count'] != 0)]

                    tempresult.append(tempLessPoint['seg'].values[0])
                tempresult.append(self._input_origi_score_list_dataframe[f].max())
                print(tempresult)
                self._seg_origi_standlist[f] = tempresult
        if self._generate_origi_score_list_way.lower() == 'the_smallest_bigger_than_or_equal':
            for f in self.__segFields:
                tempresult = []
                # tempresult.append(self._segOrigiMin.get(f))# 最小的放进来  段点是0-150 或者0-100的情况
                tempresult.append(self._input_origi_score_list_dataframe[f].min())

                for i in range(len(cussum) - 1):
                    tempMore = self._input_dataframe[self._input_dataframe[f+'_percent'] > cussum[i]]
                    tempMorePoint = tempMore[
                        (tempMore[f + '_percent'] == tempMore[f + '_percent'].min()) & (tempMore[f + '_count'] != 0)]

                    tempresult.append(tempMorePoint['seg'].values[0])
                tempresult.append(self._input_origi_score_list_dataframe[f].max())
                print(tempresult)
                self._seg_origi_standlist[f] = tempresult




def with_open(file_path):
    try:
        if not os.path.exists(file_path):
            print(file_path + "不存在")
            return
        with open(file_path, encoding='ANSI') as f:
            df = pd.read_csv(f)
            return df
    except Exception as e:
        print(e)

def outputresult(year,kl,which_point,change_way):
    expdf = with_open("D://pythontest//"+year+kl+".txt")
    seg = pg.SegTable()
    flist=[]
    if kl=='wz':
        flist=['yw', 'sx', 'wy', 'dl', 'zz', 'ls']
    if kl=='lz':
        flist = ['yw','sx','wy','wl','hx','sw']
    seg.set_data(expdf, flist)
    if kl == 'wz':
        seg.roundall(['dl', 'zz', 'ls'])
    print(seg.input_data)
    seg.set_parameters(segstep=1, segmax=150, segmin=0, segsort='ascending', usealldata=True, display=True)
    seg.run()
    seg.output_data.to_csv("D://pythontest//out"+year+kl+"_yfyd.csv",
                           index=False, header=True, mode='w')
    # output_data=with_open("D://pythontest//2015wz.csv")
    changeScore = ChangeScore()
    changeScore.set_data_yfyd(seg.output_data, flist)#设置一分一段表
    changeScore.set__input_origi_score_list_dataframe(seg.input_data) #设置需转换的原始分数表
    changeScore.set_parameters(generate_origi_score_list_way=change_way) #设置比例分数生成方式
    # the_biggest_smaller_than_or_equal
    # the_smallest_bigger_than_or_equal
    # near

    changeScore.generate_OrigiScoreList()
    changeScore.generate_stand_score(which_point)
    changeScore.output_data.to_csv("D://pythontest//out"+year+kl+"_stand.csv",
                                   index=False, header=True, mode='w')
if __name__ == '__main__':
    # expdf = with_open("D://pythontest//2015wzfkmcj.txt")
    # seg = pg.SegTable()
    #
    # seg.set_data(expdf, ['yw','sx','wy','dl','zz','ls'])
    # seg.roundall(['dl','zz','ls'])
    # print(seg.input_data)
    # seg.set_parameters(segstep=1, segmax=150, segmin=0, segsort='ascending', segalldata=True, display=True)
    # seg.run()
    # seg.output_data.to_csv("D://pythontest//2015wz.csv",
    #                        index=False, header=True, mode='w')
    # # output_data=with_open("D://pythontest//2015wz.csv")
    # changeScore = ChangeScore()
    # changeScore.set_data(seg.output_data,['yw','sx','wy','dl','zz','ls'])
    #
    # changeScore.generate_OrigiScoreList()
    # changeScore.set__input_origi_score_list_dataframe(seg.input_data)
    # changeScore.generate_stand_score()
    # changeScore.output_data.to_csv("D://pythontest//2015wzstand.csv",
    #                        index=False, header=True, mode='w')

    #   测试
    outputresult('2015','lz',0,'near')

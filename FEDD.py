import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import pacf
from scipy.stats import skew, kurtosis, pearsonr
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

''' Sources:
Cavalcante, R. C., Minku, L. L., & Oliveira, A. L. I. (2016). FEDD: Feature Extraction for Explicit Concept Drift Detection in time series. In 2016 International Joint Conference on Neural Networks (IJCNN): 24-29 July 2016, Vancouver, Canada (pp. 740–747). Piscataway, NJ: IEEE. https://doi.org/10.1109/IJCNN.2016.7727274

Ross, G. J., Adams, N. M., Tasoulis, D. K., & Hand, D. J. (2012). Exponentially weighted moving average charts for detecting concept drift. Pattern Recognition Letters, 33(2), 191–198. https://doi.org/10.1016/j.patrec.2011.08.019'''

class FeatureExtractionDriftDetector(BaseDriftDetector):
    def __init__(self, min_instances, warning_threshold, drift_threshold): 
        super().__init__()
        self.min_instances = min_instances
        self.warning_threshold=warning_threshold
        self.drift_threshold=drift_threshold
        self.warning = None
        self.warning_count=0
        self.df_stream_tot=pd.DataFrame(columns=['y', 'y_diff'])
        self.df_stream_win=pd.DataFrame(columns=['y', 'y_diff'])
        self.df_statistic=pd.DataFrame(columns=['dist', 'dist_mean', 'ewma', 'dist_std'])
        self.current_concept=None
        self.initializing=True
    
    def differencing(self, x):
        if self.df_stream_tot.empty:
            return np.nan
        else:
            return x-self.df_stream_tot.iloc[-1].y     
    
    def autocorrelation(self, data, lag):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
        
        if temp.isnull().values.any():
            return np.nan
        else:
            return temp.autocorr(lag) 
        
    def part_autocorr(self, data, lag):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
            
        if (temp.isnull().values.any()) | (len(temp)<=lag):
            return np.nan
        else:
            try:
                p_auto=pacf(temp, nlags=lag)[lag]
                
            except np.linalg.LinAlgError as e: 
                print('Warning: LinAlgError in part_autocorr function: ',e) 
                return 0 # or np.nan, 0 might distort the test statistic but nan can lead to interruption
            
            return p_auto
    
    def variance(self, data):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
            
        if temp.isnull().values.any():
            return np.nan
        else:
            return np.var(temp)
            
    def skewness(self, data):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
            
        if temp.isnull().values.any():
            return np.nan
        else:
            return skew(temp)
        
    def kurtosis(self, data):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
            
        if temp.isnull().values.any():
            return np.nan
        else:
            return kurtosis(temp)
        
    def turning_points_rate(self, data):
        if self.initializing:
            temp=data.iloc[1:].y.reset_index(drop=True) # remove first row of nans due to differencing, turning points for original values
        else:
            temp=data.y.reset_index(drop=True)
            
        if temp.isnull().values.any():
            return np.nan
        else:
            idx_max, idx_min = [], []
            if (len(temp) < 3): 
                return 0

            NEUTRAL, RISING, FALLING = range(3)
            def get_state(a, b):
                if a < b: return RISING
                if a > b: return FALLING
                return NEUTRAL

            ps = get_state(temp[0], temp[1])
            begin = 1
            for i in range(2, len(temp)):
                s = get_state(temp[i - 1], temp[i])
                if s != NEUTRAL:
                    if ps != NEUTRAL and ps != s:
                        if s == FALLING: 
                            idx_max.append((begin + i - 1) // 2)
                        else:
                            idx_min.append((begin + i - 1) // 2)
                    begin = i
                    ps = s
            return (len(idx_min) + len(idx_max))/len(temp)
    
    '''see original code in Matlab: https://de.mathworks.com/matlabcentral/fileexchange/27561-measures-of-analysis-of-time-series-toolkit-mats'''
    def bicorrelation (self, data, lag):
        if self.initializing:
            temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
        else:
            temp=data.y_diff
        
        if ((temp.isnull().values.any()) | (len(temp)<lag*2+1)):
            return np.nan
        else:
            n=len(temp)
            temp=(temp- np.mean(temp))/np.std(temp)                  
            
            return  np.sum(temp[2*lag:n].values*temp[lag:n-lag].values*temp[:n-2*lag].values)/(n-2*lag)
        
        
        '''see original code in Matlab: https://de.mathworks.com/matlabcentral/fileexchange/27561-measures-of-analysis-of-time-series-toolkit-mats'''
    def mutual_information (self, data, lag):
            
            if self.initializing:
                temp=data.iloc[1:].y_diff # remove first row of nans due to differencing
            else:
                temp=data.y_diff
                
            temp=temp.reset_index(drop=True) # prevent index issues
                
            n=len(temp)
            
            partitions=int(round(np.sqrt(n/5))) #default definition in Matlab
            if partitions is np.nan: #prevent nans
                partitions=1                
            
            if ((temp.isnull().values.any()) | (n <= lag)): # in matlab: n<2*partitions
                return np.nan

            else:
                # normalise the data
                max_temp=max(temp)
                max_ind=[i for i, j in enumerate(temp) if j == max_temp]
                min_temp=min(temp)
                temp[max_ind]=max_temp + (max_temp - min_temp)*10**(-10) # To avoid multiple exact maxima

                if (max_temp==min_temp): # prevent division by zero
                    if max_temp>0:
                        temp[:]=1
                    else:
                        temp[:]=0
                else:
                    temp=(temp-min_temp) / (max_temp-min_temp)          

                arrayV=temp*partitions
                arrayV=[math.floor(float(x))+1 for x in arrayV]
                    
                for ind in max_ind:
                    arrayV[ind]=partitions

                h1V = np.zeros((partitions,1))
                h2V = np.zeros((partitions,1))            

                ntotal=n-lag
                mutS=0
                h12M=np.zeros((partitions, partitions))

                for i in range(partitions):
                    for j in range(partitions):
                        list1=np.array([(x==i+1) for x in arrayV[lag:n]]) 
                        list2= np.array([(x==j+1) for x in arrayV[:n-lag]])
                        h12M[i,j]=np.count_nonzero(list1 & list2)

                for i in range(partitions):
                    h1V[i] = np.sum(h12M[i,:])
                    h2V[i] = np.sum(h12M[:,i])

                for i in range(partitions):
                    for j in range(partitions):
                        if h12M[i,j] >0 :
                            mutS = mutS + (h12M[i,j] / ntotal)* math.log(h12M[i,j]* ntotal / (h1V[i]*h2V[j]))

                return  mutS
        
        
    def reset(self):
        self.in_warning_zone =False
        self.warning= None
        self.warning_count=0
        self.df_stream_tot=pd.DataFrame(columns=['y', 'y_diff'])
        self.df_stream_win=pd.DataFrame(columns=['y', 'y_diff'])
        self.df_statistic=pd.DataFrame(columns=['dist', 'dist_mean', 'ewma', 'dist_std'])
        self.current_concept=None
        self.initializing=True
        
        
    def add_stream(self, x):
        if self.df_stream_tot.empty: # for first initiation
            self.df_stream_tot.loc[len(self.df_stream_tot)]=[x, self.differencing(x)]
        else:
            self.df_stream_tot.loc[len(self.df_stream_tot)]=[x, self.differencing(x)]
            
    def add_window(self, x):
        if self.df_stream_win.empty: # for first initiation
            self.df_stream_win.loc[len(self.df_stream_win)]=[x, self.differencing(x)]
        else:
            self.df_stream_win.loc[len(self.df_stream_win)]=[x, self.differencing(x)]
            
    def compute_concept(self, window):
        concept= [self.autocorrelation(window, 1), 
                  self.autocorrelation(window, 2), 
                  self.autocorrelation(window, 3),
                  self.autocorrelation(window, 4), 
                  self.autocorrelation(window, 5), 
                  self.part_autocorr(window, 1), 
                  self.part_autocorr(window, 2),
                  self.part_autocorr(window, 3), 
                  self.part_autocorr(window, 4), 
                  self.part_autocorr(window, 5),
                  self.variance(window), 
                  self.skewness(window), 
                  self.kurtosis(window), 
                  self.turning_points_rate(window), 
                  self.bicorrelation(window, 1), 
                  self.bicorrelation(window, 2), 
                  self.bicorrelation(window, 3), 
                  self.mutual_information(window, 1),
                  self.mutual_information(window, 2),
                  self.mutual_information(window, 3) ]
        return concept
        
    def compute_stat(self, dist, ind):         
        self.df_statistic.loc[ind, 'dist']=dist
        dist_mean= self.df_statistic['dist'].mean()
        self.df_statistic.loc[ind, 'dist_mean']= dist_mean

        alpha=0.2 # alpha value specified in sources
        ewma= self.df_statistic['dist'].ewm(alpha = alpha, adjust=False).mean() 
        self.df_statistic['ewma']= ewma
                
        dist_std= np.sqrt((alpha / (2-alpha)) * (1- ((1-alpha)**(2*(len(self.df_stream_win)-
                                                                    1)))))*self.df_statistic['dist'].std()
        self.df_statistic.loc[ind, 'dist_std']= dist_std
            
        
    def add_element(self, x):
        self.in_concept_change = False

        self.add_window(x)
        #print(self.df_stream_win)
        #print(len(self.df_stream_win))
        self.add_stream(x)
        
        # if window is not filled yet, do nothing
        if(len(self.df_stream_win) < self.min_instances):
            return None
        
        # if window is filled, compute features and concept
        if(len(self.df_stream_win) == self.min_instances):
            self.current_concept= self.compute_concept(self.df_stream_win.copy())
            self.initializing=False
        
        # if window is filled plus new data instance       
        if(len(self.df_stream_win) > self.min_instances):
            # limit data stream window to min_instances
            self.df_stream_win=self.df_stream_win.iloc[-self.min_instances:,:].reset_index(drop=True)
            
            # compute new concept
            new_concept= self.compute_concept(self.df_stream_win.copy())
            
            # compute pearson distance between current and new concept
            dist= 1-pearsonr(self.current_concept, new_concept)[0]
            
            # add and compute test statistics
            ind=len(self.df_statistic)
            self.compute_stat(dist, ind)
            
            new_stat=self.df_statistic.iloc[-1]
            #print(new_stat)
            
            '''warning detection'''
                
            if ((self.warning is None)& (new_stat.ewma > (new_stat.dist_mean + self.warning_threshold * 
                                                          new_stat.dist_std))):
                    self.warning = len(self.df_stream_tot)-1
                    self.in_warning_zone = True

            if (new_stat.ewma > (new_stat.dist_mean + self.warning_threshold * new_stat.dist_std)):
                    self.warning_count=0
                    self.in_warning_zone = True

            if ((not self.warning is None) & (new_stat.ewma <= (new_stat.dist_mean + self.warning_threshold *
                                                                   new_stat.dist_std))):
                    self.warning_count+=1
                    self.in_warning_zone = True

                    if self.warning_count==10: # 10 subsequent 'non-warnings' needed for reset
                        self.warning=None
                        self.warning_count=0
                        self.in_warning_zone = False

            
            '''drift detection'''
                
            if (new_stat.ewma > (new_stat.dist_mean + self.drift_threshold * new_stat.dist_std)):

                self.in_concept_change = True
                
                # reset window, starting with first warning instance
                self.df_stream_win=self.df_stream_tot.iloc[self.warning:,:]
                #print(self.df_stream_win)
                
                # if warning window is at least as long as initiation window, compute new concept
                if len(self.df_stream_win) >= self.min_instances:
                    self.current_concept= self.compute_concept(self.df_stream_win.copy())
                    
                # else delete current concept and wait for window to fill
                else:
                    self.current_concept=None
                    
                
                # reset variables
                self.in_warning_zone =False
                self.warning= None
                self.warning_count=0
                self.df_statistic=pd.DataFrame(columns=['dist', 'dist_mean', 'ewma', 'dist_std'])

            
import numpy as np
import os

class DataLoader:
    def __init__( self , DataPath , int2type = None , type2int  =None):
        self.datapath = DataPath
        self.datas = [ ]
        self.labels = [ ]
        self.onehotlabel = [ ]
        self.int2type = int2type
        self.type2int = type2int

    def dataloader(self):
        self.load_labeled_data( ) 
        Label_train, Label_test,Data_train,Data_test= self.spilt_train_test( )
        return Label_train, Label_test,Data_train,Data_test

    def load_labeled_data(self):
        audio_list = os.listdir(self.datapath)
        for i in audio_list:
            self.labels.append(i[0:5])
        self.datas = [os.path.join(self.datapath, _) for _ in audio_list] 

    def spilt_train_test( self , train_rate = 0.5 , seed = 42):
        if seed:
            np.random.seed(seed)
        shuffle_indexes  = np.random.permutation(len(self.datas))
        train_size = int( train_rate * len(self.datas))
        train_indexes = shuffle_indexes[:train_size]
        test_indexes = shuffle_indexes[train_size:]
        self.onehotlabel = self.onehot()
        Label_train = self. shuffle_label(train_indexes)
        Label_test = self. shuffle_label(test_indexes)
        Data_train = self.shuffle_data(  train_indexes)
        Data_test = self.shuffle_data(  test_indexes)
        return Label_train, Label_test,Data_train,Data_test

    def shuffle_data( self , indexes):
        new_data = [ ]
        for _ in indexes:
            new_data.append(self.datas[int(_)])
        return new_data

    def shuffle_label(self, indexes):
        new_data = []
        for _ in indexes:
            new_data.append(self.onehotlabel[int(_)])
        return new_data

    def onehot ( self ):
        self.label_type =  list(set(self.labels)) 
        if self.int2type == None:
            self.type2int = dict( ( c , i ) for i,c in enumerate( self.label_type ) )
            self.int2type = dict( ( i , c ) for i,c in enumerate( self.label_type ) )
        integer_encoded = [ self.type2int[ type ] for type in self.labels ]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(self.label_type))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

class BalanceDataLoad_(DataLoader):
    def __init__( self , DataPath, int2type = None , type2int  =None):
        self.datapath = DataPath
        self.train_datas = [ ]
        self.train_labels = [ ]
        self.test_datas = [ ]
        self.test_labels = [ ]
        self.onehot_test_label = [ ]
        self.onehot_train_label = [ ]
        self.int2type = int2type
        self.type2int = type2int

    def dataloader(self):
        self.load_labeled_data( )
        return self.onehot_train_label,self.onehot_test_label,self.train_datas,self.test_datas

    def load_labeled_data(self):
        folders_list = os.listdir(self.datapath)
        for folder_label in folders_list:
            folderpath=os.path.join(self.datapath, folder_label)
            audio_list = os.listdir(folderpath)
            audio_paths = [os.path.join(folderpath, _) for _ in audio_list] 
            shuffle_audio_indexes  = np.random.permutation(len(audio_list))
            for audio_number in range( 0 , 80 , 2 ):
                self.load_label ( self.test_labels , folder_label )
                self.load_data ( self.test_datas , audio_paths[shuffle_audio_indexes[audio_number]])
                self.load_label ( self.train_labels , folder_label )
                self.load_data ( self.train_datas , audio_paths[shuffle_audio_indexes[audio_number+1]])    
        type2int,int2type,self.onehot_test_label , self.onehot_train_label = self.onehot()
        print(type2int,int2type)
        self.shuffling()

    def shuffling(self):
        shuffle_data_indexes  = np.random.permutation(len(self.train_datas))
        self.train_datas = self.shuffle_data(self.train_datas,shuffle_data_indexes)
        self.onehot_train_label = self.shuffle_data(self.onehot_train_label,shuffle_data_indexes)

    def shuffle_data( self , datas , indexes):
        new_data = [ ]
        for _ in indexes:
            new_data.append(datas[_])
        return new_data

    def onehot ( self ):
        self.label_type =  list(set(self.test_labels))  
        if self.int2type == None:
            self.type2int = dict( ( c , i ) for i,c in enumerate( self.label_type ) )
            self.int2type = dict( ( i , c ) for i,c in enumerate( self.label_type ) )
        test_integer_encoded = [ self.type2int[ type ] for type in self.test_labels ]
        train_integer_encoded = [ self.type2int[ type ] for type in self.train_labels ]
        onehot_test_encoded = self.onehot_encode(test_integer_encoded)
        onehot_train_encoded = self.onehot_encode(train_integer_encoded)
        return self.type2int,self.int2type,onehot_test_encoded,onehot_train_encoded

    def onehot_encode(self, integer_encoded):
        onehot_encoded = list () 
        for value in integer_encoded:
            letter = [0 for _ in range(len(self.label_type))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    def load_label ( self , labels , folder_label ):
        labels.append(folder_label)

    def load_data ( self , data , path):
        data.append(path)

class NoiseLoader(DataLoader):
    def __init__( self , DataPath,int2type = None , type2int  =None):
        self.datapath = DataPath
        self.datas = [ ]
        self.labels = [ ]
        self.onehot_label = [ ]
        self.int2type = int2type
        self.type2int = type2int

    def dataloader(self):
        self.load_labeled_data( )
        return self.onehot_label,self.datas

    def load_labeled_data(self):
        folders_list = os.listdir(self.datapath)
        for folder_label in folders_list:
            folderpath=os.path.join(self.datapath, folder_label)
            audio_list = os.listdir(folderpath)
            audio_paths = [os.path.join(folderpath, _) for _ in audio_list] 
            for audio_number in range( len(audio_list) ):
                self.load_label ( self.labels , folder_label )
                self.load_data ( self.datas , audio_paths[audio_number]) 
        self.onehot_label = self.onehot()
        print(self.type2int,self.int2type)


    def onehot ( self ):
        self.label_type =  list(set(self.labels)) 
        if self.int2type == None:
            self.type2int = dict( ( c , i ) for i,c in enumerate( self.label_type ) )
            self.int2type = dict( ( i , c ) for i,c in enumerate( self.label_type ) )
        integer_encoded = [ self.type2int[ type ] for type in self.labels ]
        onehot_encoded = self.onehot_encode(integer_encoded)
        return onehot_encoded

    def onehot_encode(self, integer_encoded):
        onehot_encoded = list () 
        for value in integer_encoded:
            letter = [0 for _ in range(len(self.label_type))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    def load_label ( self , labels , folder_label ):
        labels.append(folder_label)

    def load_data ( self , data , path):
        data.append(path)
import numpy as np

def add_3_of_a_kind(picture, code, file, class_number):
        class_array = np.array([class_number,class_number,class_number])
        class_array = np.expand_dims(np.expand_dims(class_array, axis=0),axis = 2)
        
        diff_classes = np.array([1], dtype=np.int)
        diff_classes = np.expand_dims(diff_classes, axis=0)
        
        files = np.expand_dims(np.expand_dims(file,axis=0),axis=2)
        
        images = np.expand_dims(picture,axis=0)
        
        codes = np.expand_dims(code,axis=0)
        
        return images, codes, files, class_array, diff_classes
    
    
def add_2_1_of_a_kind(picture, code, file, class_number, picture_1, code_1, file_1, class_number_1):
    
        class_array = np.array([class_number,class_number,class_number_1])
        class_array = np.expand_dims(np.expand_dims(class_array, axis=0),axis = 2)
        
        diff_classes = np.array([2], dtype=np.int)
        diff_classes = np.expand_dims(diff_classes, axis=0)
        
        index = np.array([0,1])
        index_1 = np.array([2])
        
        file = file[index]
        file_1 = file_1[index_1]     
        file = np.append(file, file_1, axis=0)    
        files = np.expand_dims(np.expand_dims(file,axis=0),axis=2)
        
        picture = picture[index]
        picture_1 = picture_1[index_1]
        picture = np.append(picture, picture_1, axis=0) 
        images = np.expand_dims(picture,axis=0)
        
        
        code = code[index]
        code_1 = code_1[index_1]
        code = np.append(code,code_1, axis=0)
        codes = np.expand_dims(code,axis=0)
        
        return images, codes, files, class_array, diff_classes
    
def add_1_2_of_a_kind(picture, code, file, class_number, picture_1, code_1, file_1, class_number_1):
    
        class_array = np.array([class_number,class_number_1,class_number_1])
        class_array = np.expand_dims(np.expand_dims(class_array, axis=0),axis = 2)
        
        diff_classes = np.array([2], dtype=np.int)
        diff_classes = np.expand_dims(diff_classes, axis=0)
        
        index = np.array([0])
        index_1 = np.array([1,2])
        
        file = file[index]
        file_1 = file_1[index_1]     
        file = np.append(file, file_1, axis=0)    
        files = np.expand_dims(np.expand_dims(file,axis=0),axis=2)
        
        picture = picture[index]
        picture_1 = picture_1[index_1]
        picture = np.append(picture, picture_1, axis=0) 
        images = np.expand_dims(picture,axis=0)
        
        
        code = code[index]
        code_1 = code_1[index_1]
        code = np.append(code,code_1, axis=0)
        codes = np.expand_dims(code,axis=0)
        
        return images, codes, files, class_array, diff_classes

def add_3_different(picture, code, file, class_number, picture_1, code_1, file_1, class_number_1, picture_2, code_2, file_2, class_number_2):
        class_array = np.array([class_number,class_number_1,class_number_2])
        class_array = np.expand_dims(np.expand_dims(class_array, axis=0),axis = 2)
        
        diff_classes = np.array([3], dtype=np.int)
        diff_classes = np.expand_dims(diff_classes, axis=0)
        
        index = np.array([0])
        index_1 = np.array([1])
        index_2 = np.array([2])
        
        file = file[index]
        file_1 = file_1[index_1]
        file_2 = file_2[index_2]
        file = np.append(file, file_1, axis=0)
        file = np.append(file, file_2, axis=0)
        files = np.expand_dims(np.expand_dims(file,axis=0),axis=2)
        
        picture = picture[index]
        picture_1 = picture_1[index_1]
        picture_2 = picture_2[index_2]
        picture = np.append(picture, picture_1, axis=0)
        picture = np.append(picture, picture_2, axis=0)
        images = np.expand_dims(picture,axis=0)
        
        
        code = code[index]
        code_1 = code_1[index_1]
        code_2 = code_2[index_2]
        code = np.append(code,code_1, axis=0)
        code = np.append(code,code_2, axis=0)
        codes = np.expand_dims(code,axis=0)
        
        return images, codes, files, class_array, diff_classes
    
def append_set(images, codes, files, class_array, diff_classes, image_set_array, code_set_array, file_set_array, correct_class_array, number_different_classes):
        #Put together
        image_set_array = np.append(image_set_array, images, axis=0)
        code_set_array = np.append(code_set_array, codes, axis = 0)
        file_set_array = np.append(file_set_array, files, axis = 0)
        correct_class_array = np.append(correct_class_array, class_array, axis = 0)
        number_different_classes = np.append(number_different_classes, diff_classes, axis=0)
        
        return image_set_array, code_set_array, file_set_array, correct_class_array, number_different_classes
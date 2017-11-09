def set_model_name(model_name, target_dir):
    '''function to set default model name'''
    
    if model_name == None:  
        return model_dir+os.sep+ 'model.hdf5'
    return  model_dir+os.sep+splitext(basename(model_name))[0]+'.hdf5'

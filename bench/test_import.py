
import importlib
import sys
import os

# Add workspace root to sys.path
sys.path.append(os.getcwd())

blobs = [
    'nirs4all.operators.transforms.scalers.StandardNormalVariate',
    'nirs4all.operators.transforms.nirs.SavitzkyGolay'
]

for blob in blobs:
    print(f'\nTesting {blob}')
    mod_name, _, cls_or_func_name = blob.rpartition('.')

    try:
        mod = importlib.import_module(mod_name)
        print(f'Module imported: {mod}')
        cls_or_func = getattr(mod, cls_or_func_name)
        print(f'Class found: {cls_or_func}')
        instance = cls_or_func()
        print(f'Instance created: {instance}')
        
        if hasattr(instance, 'get_params'):
            print(f'get_params: {instance.get_params()}')
        else:
            print('get_params NOT found')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()


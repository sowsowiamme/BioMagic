class Singleton:
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            print("Singleton initialized")
            for args in args:
                setattr(self, args, args)
            for kwargs in kwargs:
                setattr(self, kwargs, kwargs)


s1 = Singleton("name2", "age2")
s2 = Singleton("name2", "age2")

print(s1.name2, s1.age2)
print(s2.name2, s2.age2)
# __new__ is called before __init__
# __new__ is used to create the instance, and it is a class method
# __init__ is used to initialize the instance, and it is an instance/object method
def singleton_decorator(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        print("当前仓库里的内容是：", instances)
        return instances[cls]
    return get_instance

@singleton_decorator
class Singleton:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __str__(self):
        return f"Singleton({self.args}, {self.kwargs})"
@singleton_decorator
class DataBase:
    pass

@singleton_decorator
class Logger:
    pass

db1 = DataBase()
logger1 = Logger()




def singleton(cls):
    # 把 instances 变成 singleton 函数自己的属性，这样它就只会被创建一次
    if not hasattr(singleton, 'instances'):
        singleton.instances = {}
        
    def get_instance(*args, **kwargs):
        if cls not in singleton.instances:
            singleton.instances[cls] = cls(*args, **kwargs)
            
        # 打印看看大仓库里有什么
        print(f"当前大仓库里的内容: {singleton.instances}")
        return singleton.instances[cls]
        
    return get_instance

# ================= 测试一下 =================

@singleton
class DataBase:
    pass

@singleton
class Logger:
    pass

print("--- 实例化 DataBase ---")
db1 = DataBase()

print("\n--- 实例化 Logger ---")
log1 = Logger()

print("\n--- 再次实例化 DataBase ---")
db2 = DataBase()


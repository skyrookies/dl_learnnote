# CPP learning 12.17

### class 声明结构

```c++
class name//class head
{
//class body
public:
  
private:

};
```

### 防御式头文件结构

```c++
#ifndef complex_test_hpp
#define complex_test_hpp

#include <stdio.h>
//填充内容
#endif 
/* complex_test_hpp */
/*
#ifndef 
...
#endif */
//让头文件在include时自动检查是不是第一次使用，不会造成多次include或者include顺序错误
```

### class 模版() 

```c++
template<typename T>
class ClassName
{

};
{
	ClassName<double> c1(1,2.5);
	ClassName<ubt> c2(1,2);
}
```

### inline 函数 内联函数

定义：若函数在class body内定义，成为inline function的候选者，最终是否称为内联函数由编译器决定，一般来说若过于复杂则不作为内联函数。内联函数运行速度较快 

### access level

访问级别，即public 和 private
一般来说 内部数据都放在private 函数需要被外界调用放在public

第三种访问级别为 protected

写作顺序可以任意交错

### constructor(ctor) 构造函数

构造函数

```cpp
class complex
{
  public:
  	complex(double r = 0 , double i = 0 ):re(r),im(r) {}
/*
1 构造函数没有返回值 返回值就是构造本身
2 构造函数函数名与class名 保持一致
3 构造函数使用过程包括 initialization list 初始值过程和之后{}中的函数过程，而一般初始值赋值即可在初始值过程中传递也可在函数过程中赋值，但是一般在initialization list 中传递更规范
4 构造函数也可以有很多overloading 但是注意不可使用函数构造过程中会使函数构造混淆的重构
5 对于Object Based class构建 一般情况下不需要析构函数
*/
  private:
  	double re,im;
}
```



###  
# CPP learning 12.19

### 构造函数在private区的使用

一般来说 构造函数的访问级别access level不应在private但是在不允许被外界调用的情况下可以使用
例如设计模式中最简单的其中之一 ： Singleton 模式

### const member function 常量成员函数

在函数后写入，表示该函数不会改变数据内容

```c++
double function_name () const {   	;}
```

### 注意：

若未使用常量成员关键字const的参数 在后续应用中若实例化过程中使用const 

```
	const complex c1(1,2);
```

则编译器会报错 
故应养成习惯，在写类中方法function时 该写const 尽量写上

### 参数传递 pass by value / pass by reference (to const)

参数传递 首先考虑引用reference传递，效率更高，只需传递四个字节地址。
在cpp中 reference 引用的本质仍然是指针，但是比指针多了一些好用的特性，比如下面提到到的 

### 常量引用传递

即传递引用前加 const 关键字 表示 该传递的引用地址不改变地址内数据具体的值

```cpp
class complex
{
public:
  complex (double r = 0, double i = 0): re (r), im (i) { }
  complex& operator += (const complex&);
//上一行操作中(const complex&) 即对complex对象的引用中 const 表示不改变该实例具体的值
}
```

### 返回值传递 return by value / return by reference (to const)

同传参

### 友元 friend

友元可以自由取得friend的private成员

而class的各个objects互为友元friends


жє	
/ѕ.
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.10.02
b'unknown'є

initNoOp

init_1NoOp
z
input_1Placeholder*$
shape:џџџџџџџџџdd*/
_output_shapes
:џџџџџџџџџdd*
dtype0
v
conv2d_1/random_uniform/shapeConst*
_output_shapes
:*%
valueB"             *
dtype0
`
conv2d_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *їќSН*
dtype0
`
conv2d_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *їќS=*
dtype0
В
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*&
_output_shapes
: *
T0*
seedБџх)*
seed2дрр*
dtype0
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 

conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
: 

conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*
T0*&
_output_shapes
: 

conv2d_1/kernel
VariableV2*
	container *
shared_name *
shape: *&
_output_shapes
: *
dtype0
Ш
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
use_locking(*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: *
validate_shape(

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: 
[
conv2d_1/ConstConst*
_output_shapes
: *
valueB *    *
dtype0
y
conv2d_1/bias
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
­
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
use_locking(* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
: *
validate_shape(
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
: 
s
"conv2d_1/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ц
conv2d_1/convolutionConv2Dinput_1conv2d_1/kernel/read*
data_formatNHWC*
paddingVALID*
	dilations
*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:џџџџџџџџџ]] *
strides


conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:џџџџџџџџџ]] 
a
conv2d_1/TanhTanhconv2d_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ]] 
О
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Tanh*
data_formatNHWC*
paddingVALID*
ksize
*
T0*/
_output_shapes
:џџџџџџџџџ.. *
strides

v
conv2d_2/random_uniform/shapeConst*
_output_shapes
:*%
valueB"              *
dtype0
`
conv2d_2/random_uniform/minConst*
_output_shapes
: *
valueB
 *qФН*
dtype0
`
conv2d_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *qФ=*
dtype0
В
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*&
_output_shapes
:  *
T0*
seedБџх)*
seed2МЛ*
dtype0
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 

conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
:  

conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*
T0*&
_output_shapes
:  

conv2d_2/kernel
VariableV2*
	container *
shared_name *
shape:  *&
_output_shapes
:  *
dtype0
Ш
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:  *
validate_shape(

conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:  
[
conv2d_2/ConstConst*
_output_shapes
: *
valueB *    *
dtype0
y
conv2d_2/bias
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
­
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
use_locking(* 
_class
loc:@conv2d_2/bias*
T0*
_output_shapes
: *
validate_shape(
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
T0*
_output_shapes
: 
s
"conv2d_2/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
і
conv2d_2/convolutionConv2Dmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
data_formatNHWC*
paddingVALID*
	dilations
*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:џџџџџџџџџ'' *
strides


conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
data_formatNHWC*
T0*/
_output_shapes
:џџџџџџџџџ'' 
a
conv2d_2/TanhTanhconv2d_2/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ'' 
О
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Tanh*
data_formatNHWC*
paddingVALID*
ksize
*
T0*/
_output_shapes
:џџџџџџџџџ *
strides

f
flatten_1/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
T0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
i
flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Џ
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*

begin_mask *
Index0*
new_axis_mask *
shrink_axis_mask *
ellipsis_mask *
T0*
_output_shapes
:*
end_mask
Y
flatten_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
\
flatten_1/stack/0Const*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
N*
T0*
_output_shapes
:*

axis 

flatten_1/ReshapeReshapemax_pooling2d_2/MaxPoolflatten_1/stack*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0
m
dense_1/random_uniform/shapeConst*
_output_shapes
:*
valueB" -     *
dtype0
_
dense_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *ttКМ*
dtype0
_
dense_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *ttК<*
dtype0
Љ
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
_output_shapes
:	 Z*
T0*
seedБџх)*
seed2кЇт*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	 Z

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	 Z

dense_1/kernel
VariableV2*
	container *
shared_name *
shape:	 Z*
_output_shapes
:	 Z*
dtype0
Н
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	 Z*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	 Z
Z
dense_1/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_1/bias
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:

dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
]
dense_1/SigmoidSigmoiddense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
_
dense_2/random_uniform/minConst*
_output_shapes
: *
valueB
 *шЁО*
dtype0
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *шЁ>*
dtype0
Ј
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes

:*
T0*
seedБџх)*
seed2ВкЫ*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 

dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:

dense_2/kernel
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
М
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(
{
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:
Z
dense_2/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_2/bias
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
_class
loc:@dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
T0*
_output_shapes
:

dense_2/MatMulMatMuldense_1/Sigmoiddense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
]
dense_2/SigmoidSigmoiddense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
m
dense_3/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
_
dense_3/random_uniform/minConst*
_output_shapes
: *
valueB
 *зГнО*
dtype0
_
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *зГн>*
dtype0
Ј
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
_output_shapes

:*
T0*
seedБџх)*
seed2Њ*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 

dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:

dense_3/kernel
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
М
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes

:*
validate_shape(
{
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes

:
Z
dense_3/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_3/bias
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:*
validate_shape(
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:

dense_3/MatMulMatMuldense_2/Sigmoiddense_3/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
]
dense_3/SigmoidSigmoiddense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
_
Adam/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
s
Adam/iterations
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0	
О
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*"
_class
loc:@Adam/iterations*
T0	*
_output_shapes
: *
validate_shape(
v
Adam/iterations/readIdentityAdam/iterations*"
_class
loc:@Adam/iterations*
T0	*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
_output_shapes
: *
valueB
 *o:*
dtype0
k
Adam/lr
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
_class
loc:@Adam/lr*
T0*
_output_shapes
: *
validate_shape(
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
T0*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
o
Adam/beta_1
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
Ў
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
_class
loc:@Adam/beta_1*
T0*
_output_shapes
: *
validate_shape(
j
Adam/beta_1/readIdentityAdam/beta_1*
_class
loc:@Adam/beta_1*
T0*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*
dtype0
o
Adam/beta_2
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
Ў
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
_class
loc:@Adam/beta_2*
T0*
_output_shapes
: *
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
T0*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
n

Adam/decay
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
Њ
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
_class
loc:@Adam/decay*
T0*
_output_shapes
: *
validate_shape(
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
T0*
_output_shapes
: 

dense_3_targetPlaceholder*%
shape:џџџџџџџџџџџџџџџџџџ*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0
q
dense_3_sample_weightsPlaceholder*
shape:џџџџџџџџџ*#
_output_shapes
:џџџџџџџџџ*
dtype0
r
'loss/dense_3_loss/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
Ѕ
loss/dense_3_loss/SumSumdense_3/Sigmoid'loss/dense_3_loss/Sum/reduction_indices*
	keep_dims(*
T0*

Tidx0*'
_output_shapes
:џџџџџџџџџ
~
loss/dense_3_loss/truedivRealDivdense_3/Sigmoidloss/dense_3_loss/Sum*
T0*'
_output_shapes
:џџџџџџџџџ
\
loss/dense_3_loss/ConstConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
\
loss/dense_3_loss/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
o
loss/dense_3_loss/subSubloss/dense_3_loss/sub/xloss/dense_3_loss/Const*
T0*
_output_shapes
: 

'loss/dense_3_loss/clip_by_value/MinimumMinimumloss/dense_3_loss/truedivloss/dense_3_loss/sub*
T0*'
_output_shapes
:џџџџџџџџџ

loss/dense_3_loss/clip_by_valueMaximum'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*
T0*'
_output_shapes
:џџџџџџџџџ
o
loss/dense_3_loss/LogLogloss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
u
loss/dense_3_loss/mulMuldense_3_targetloss/dense_3_loss/Log*
T0*'
_output_shapes
:џџџџџџџџџ
t
)loss/dense_3_loss/Sum_1/reduction_indicesConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
Ћ
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/mul)loss/dense_3_loss/Sum_1/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/NegNegloss/dense_3_loss/Sum_1*
T0*#
_output_shapes
:џџџџџџџџџ
k
(loss/dense_3_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
Њ
loss/dense_3_loss/MeanMeanloss/dense_3_loss/Neg(loss/dense_3_loss/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
|
loss/dense_3_loss/mul_1Mulloss/dense_3_loss/Meandense_3_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
a
loss/dense_3_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0

loss/dense_3_loss/NotEqualNotEqualdense_3_sample_weightsloss/dense_3_loss/NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
w
loss/dense_3_loss/CastCastloss/dense_3_loss/NotEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_3_loss/Mean_1Meanloss/dense_3_loss/Castloss/dense_3_loss/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 

loss/dense_3_loss/truediv_1RealDivloss/dense_3_loss/mul_1loss/dense_3_loss/Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
c
loss/dense_3_loss/Const_2Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_3_loss/Mean_2Meanloss/dense_3_loss/truediv_1loss/dense_3_loss/Const_2*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
O

loss/mul/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
V
loss/mulMul
loss/mul/xloss/dense_3_loss/Mean_2*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
i
metrics/acc/ArgMax_1/dimensionConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0

metrics/acc/ArgMax_1ArgMaxdense_3/Sigmoidmetrics/acc/ArgMax_1/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
h
metrics/acc/CastCastmetrics/acc/Equal*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
[
metrics/acc/ConstConst*
_output_shapes
:*
valueB: *
dtype0
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
_output_shapes
: *
valueB *
dtype0

!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ж
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*

index_type0*
_class
loc:@loss/mul*
T0*
_output_shapes
: 
І
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_3_loss/Mean_2*
_class
loc:@loss/mul*
T0*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
T0*
_output_shapes
: 
К
Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shapeConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
:*
valueB:*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape/shape*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
:*
Tshape0
У
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ShapeShapeloss/dense_3_loss/truediv_1*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
:*
out_type0
Ћ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ
Х
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1Shapeloss/dense_3_loss/truediv_1*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
:*
out_type0
­
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
valueB *
dtype0
В
;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
:*
valueB: *
dtype0
Љ
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const*
	keep_dims( *+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
: *

Tidx0
Д
=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1Const*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
:*
valueB: *
dtype0
­
<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Const_1*
	keep_dims( *+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
: *

Tidx0
Ў
?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/yConst*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: *
value	B :*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum/y*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
: 

>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Maximum*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*
_output_shapes
: 
п
:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/floordiv*

DstT0*

SrcT0*+
_class!
loc:@loss/dense_3_loss/Mean_2*
_output_shapes
: 

=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/Cast*+
_class!
loc:@loss/dense_3_loss/Mean_2*
T0*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ShapeShapeloss/dense_3_loss/mul_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*
_output_shapes
:*
out_type0
Г
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1Const*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
_output_shapes
: *
valueB *
dtype0
ж
Ntraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivloss/dense_3_loss/Mean_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
Х
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/SumSum@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDivNtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*
_output_shapes
:*

Tidx0
Е
@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
К
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/NegNegloss/dense_3_loss/mul_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1RealDiv<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Negloss/dense_3_loss/Mean_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ

Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2RealDivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_1loss/dense_3_loss/Mean_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
Є
<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulMul=training/Adam/gradients/loss/dense_3_loss/Mean_2_grad/truedivBtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/RealDiv_2*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
Х
>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1Sum<training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/mulPtraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
	keep_dims( *.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*
_output_shapes
:*

Tidx0
Ў
Btraining/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape_1Reshape>training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Sum_1@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Shape_1*.
_class$
" loc:@loss/dense_3_loss/truediv_1*
T0*
_output_shapes
: *
Tshape0
М
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ShapeShapeloss/dense_3_loss/Mean**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*
_output_shapes
:*
out_type0
О
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1Shapedense_3_sample_weights**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*
_output_shapes
:*
out_type0
Ц
Jtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulMul@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshapedense_3_sample_weights**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*#
_output_shapes
:џџџџџџџџџ
Б
8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/SumSum8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/MulJtraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs*
	keep_dims( **
_class 
loc:@loss/dense_3_loss/mul_1*
T0*
_output_shapes
:*

Tidx0
Ѕ
<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeReshape8training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
ѕ
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Mulloss/dense_3_loss/Mean@training/Adam/gradients/loss/dense_3_loss/truediv_1_grad/Reshape**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*#
_output_shapes
:џџџџџџџџџ
З
:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Mul_1Ltraining/Adam/gradients/loss/dense_3_loss/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( **
_class 
loc:@loss/dense_3_loss/mul_1*
T0*
_output_shapes
:*

Tidx0
Ћ
>training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Reshape_1Reshape:training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Sum_1<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/Shape_1**
_class 
loc:@loss/dense_3_loss/mul_1*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
Й
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ShapeShapeloss/dense_3_loss/Neg*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:*
out_type0
Ѕ
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B :*
dtype0
№
7training/Adam/gradients/loss/dense_3_loss/Mean_grad/addAdd(loss/dense_3_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: 

7training/Adam/gradients/loss/dense_3_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_3_loss/Mean_grad/add8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: 
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:*
valueB: *
dtype0
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B : *
dtype0
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B :*
dtype0
б
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_3_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_3_loss/Mean*

Tidx0*
_output_shapes
:
Ћ
>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/valueConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B :*
dtype0

8training/Adam/gradients/loss/dense_3_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill/value*

index_type0*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: 

Atraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_3_loss/Mean_grad/range7training/Adam/gradients/loss/dense_3_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Fill*
N*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:
Њ
=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/yConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B :*
dtype0

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum/y*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:

<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:
Ў
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/ReshapeReshape<training/Adam/gradients/loss/dense_3_loss/mul_1_grad/ReshapeAtraining/Adam/gradients/loss/dense_3_loss/Mean_grad/DynamicStitch*)
_class
loc:@loss/dense_3_loss/Mean*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
І
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv*)
_class
loc:@loss/dense_3_loss/Mean*
T0*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ
Л
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_2Shapeloss/dense_3_loss/Neg*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:*
out_type0
М
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3Shapeloss/dense_3_loss/Mean*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
:*
out_type0
Ў
9training/Adam/gradients/loss/dense_3_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:*
valueB: *
dtype0
Ё
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const*
	keep_dims( *)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: *

Tidx0
А
;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
:*
valueB: *
dtype0
Ѕ
:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_3_loss/Mean_grad/Const_1*
	keep_dims( *)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: *

Tidx0
Ќ
?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: *
value	B :*
dtype0

=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1/y*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: 

>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_3_loss/Mean_grad/Maximum_1*)
_class
loc:@loss/dense_3_loss/Mean*
T0*
_output_shapes
: 
л
8training/Adam/gradients/loss/dense_3_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_3_loss/Mean_grad/floordiv_1*

DstT0*

SrcT0*)
_class
loc:@loss/dense_3_loss/Mean*
_output_shapes
: 

;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_3_loss/Mean_grad/Cast*)
_class
loc:@loss/dense_3_loss/Mean*
T0*#
_output_shapes
:џџџџџџџџџ
в
6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegNeg;training/Adam/gradients/loss/dense_3_loss/Mean_grad/truediv*(
_class
loc:@loss/dense_3_loss/Neg*
T0*#
_output_shapes
:џџџџџџџџџ
Л
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ShapeShapeloss/dense_3_loss/mul**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
:*
out_type0
Ї
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/SizeConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
value	B :*
dtype0
ђ
8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/addAdd)loss/dense_3_loss/Sum_1/reduction_indices9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
: 

8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/modFloorMod8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/add9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
: 
Ћ
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1Const**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
valueB *
dtype0
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/startConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
value	B : *
dtype0
Ў
@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/deltaConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
value	B :*
dtype0
ж
:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/rangeRange@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/start9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Size@training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range/delta**
_class 
loc:@loss/dense_3_loss/Sum_1*

Tidx0*
_output_shapes
:
­
?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/valueConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
value	B :*
dtype0

9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/FillFill<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape_1?training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill/value*

index_type0**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
: 

Btraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitchDynamicStitch:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/range8training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/mod:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Fill*
N**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
:
Ќ
>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/yConst**
_class 
loc:@loss/dense_3_loss/Sum_1*
_output_shapes
: *
value	B :*
dtype0

<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/MaximumMaximumBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch>training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum/y**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
:

=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Shape<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Maximum**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*
_output_shapes
:
И
<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/Neg_grad/NegBtraining/Adam/gradients/loss/dense_3_loss/Sum_1_grad/DynamicStitch**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0
Ў
9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/TileTile<training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Reshape=training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/floordiv**
_class 
loc:@loss/dense_3_loss/Sum_1*
T0*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ
А
8training/Adam/gradients/loss/dense_3_loss/mul_grad/ShapeShapedense_3_target*(
_class
loc:@loss/dense_3_loss/mul*
T0*
_output_shapes
:*
out_type0
Й
:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1Shapeloss/dense_3_loss/Log*(
_class
loc:@loss/dense_3_loss/mul*
T0*
_output_shapes
:*
out_type0
О
Htraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*(
_class
loc:@loss/dense_3_loss/mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ы
6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulMul9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tileloss/dense_3_loss/Log*(
_class
loc:@loss/dense_3_loss/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
6training/Adam/gradients/loss/dense_3_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_3_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *(
_class
loc:@loss/dense_3_loss/mul*
T0*
_output_shapes
:*

Tidx0
Њ
:training/Adam/gradients/loss/dense_3_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape*(
_class
loc:@loss/dense_3_loss/mul*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0
ц
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Muldense_3_target9training/Adam/gradients/loss/dense_3_loss/Sum_1_grad/Tile*(
_class
loc:@loss/dense_3_loss/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_3_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *(
_class
loc:@loss/dense_3_loss/mul*
T0*
_output_shapes
:*

Tidx0
Ї
<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_3_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_3_loss/mul_grad/Shape_1*(
_class
loc:@loss/dense_3_loss/mul*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal
Reciprocalloss/dense_3_loss/clip_by_value=^training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1*(
_class
loc:@loss/dense_3_loss/Log*
T0*'
_output_shapes
:џџџџџџџџџ

6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulMul<training/Adam/gradients/loss/dense_3_loss/mul_grad/Reshape_1=training/Adam/gradients/loss/dense_3_loss/Log_grad/Reciprocal*(
_class
loc:@loss/dense_3_loss/Log*
T0*'
_output_shapes
:џџџџџџџџџ
н
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeShape'loss/dense_3_loss/clip_by_value/Minimum*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*
_output_shapes
:*
out_type0
Л
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1Const*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
: *
valueB *
dtype0
ю
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*
_output_shapes
:*
out_type0
С
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/ConstConst*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
_output_shapes
: *
valueB
 *    *
dtype0
в
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros/Const*

index_type0*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ

Itraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_3_loss/clip_by_value/Minimumloss/dense_3_loss/Const*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
ц
Rtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
њ
Ctraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_3_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
ќ
Etraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_3_loss/Log_grad/mul*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
д
@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*
_output_shapes
:*

Tidx0
Щ
Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
к
Btraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*
_output_shapes
:*

Tidx0
О
Ftraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Shape_1*2
_class(
&$loc:@loss/dense_3_loss/clip_by_value*
T0*
_output_shapes
: *
Tshape0
п
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeShapeloss/dense_3_loss/truediv*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0
Ы
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1Const*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
: *
valueB *
dtype0

Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0
б
Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/ConstConst*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
_output_shapes
: *
valueB
 *    *
dtype0
ђ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros/Const*

index_type0*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualloss/dense_3_loss/truedivloss/dense_3_loss/sub*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ

Ztraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
Ktraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zeros*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
Mtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_3_loss/clip_by_value_grad/Reshape*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
є
Htraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*
_output_shapes
:*

Tidx0
щ
Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
њ
Jtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*
_output_shapes
:*

Tidx0
о
Ntraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_3_loss/clip_by_value/Minimum*
T0*
_output_shapes
: *
Tshape0
Й
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/ShapeShapedense_3/Sigmoid*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*
_output_shapes
:*
out_type0
С
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1Shapeloss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*
_output_shapes
:*
out_type0
Ю
Ltraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivRealDivLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshapeloss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Н
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*
_output_shapes
:*

Tidx0
Б
>training/Adam/gradients/loss/dense_3_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
В
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/NegNegdense_3/Sigmoid*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
ў
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_3_loss/truediv_grad/Negloss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ

@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_1loss/dense_3_loss/Sum*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Б
:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulMulLtraining/Adam/gradients/loss/dense_3_loss/clip_by_value/Minimum_grad/Reshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/RealDiv_2*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Н
<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_3_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_3_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*
_output_shapes
:*

Tidx0
З
@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_3_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Б
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/ShapeShapedense_3/Sigmoid*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
:*
out_type0
Ѓ
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/SizeConst*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
value	B :*
dtype0
ъ
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/addAdd'loss/dense_3_loss/Sum/reduction_indices7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
: 
ў
6training/Adam/gradients/loss/dense_3_loss/Sum_grad/modFloorMod6training/Adam/gradients/loss/dense_3_loss/Sum_grad/add7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
: 
Ї
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1Const*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
valueB *
dtype0
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/startConst*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
value	B : *
dtype0
Њ
>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/deltaConst*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
value	B :*
dtype0
Ь
8training/Adam/gradients/loss/dense_3_loss/Sum_grad/rangeRange>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/start7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Size>training/Adam/gradients/loss/dense_3_loss/Sum_grad/range/delta*(
_class
loc:@loss/dense_3_loss/Sum*

Tidx0*
_output_shapes
:
Љ
=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/valueConst*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
value	B :*
dtype0

7training/Adam/gradients/loss/dense_3_loss/Sum_grad/FillFill:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape_1=training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill/value*

index_type0*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
: 

@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitchDynamicStitch8training/Adam/gradients/loss/dense_3_loss/Sum_grad/range6training/Adam/gradients/loss/dense_3_loss/Sum_grad/mod8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Fill*
N*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
:
Ј
<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/yConst*(
_class
loc:@loss/dense_3_loss/Sum*
_output_shapes
: *
value	B :*
dtype0

:training/Adam/gradients/loss/dense_3_loss/Sum_grad/MaximumMaximum@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch<training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum/y*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
:

;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordivFloorDiv8training/Adam/gradients/loss/dense_3_loss/Sum_grad/Shape:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Maximum*(
_class
loc:@loss/dense_3_loss/Sum*
T0*
_output_shapes
:
М
:training/Adam/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshape@training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape_1@training/Adam/gradients/loss/dense_3_loss/Sum_grad/DynamicStitch*(
_class
loc:@loss/dense_3_loss/Sum*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0
І
7training/Adam/gradients/loss/dense_3_loss/Sum_grad/TileTile:training/Adam/gradients/loss/dense_3_loss/Sum_grad/Reshape;training/Adam/gradients/loss/dense_3_loss/Sum_grad/floordiv*(
_class
loc:@loss/dense_3_loss/Sum*
T0*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ

training/Adam/gradients/AddNAddN>training/Adam/gradients/loss/dense_3_loss/truediv_grad/Reshape7training/Adam/gradients/loss/dense_3_loss/Sum_grad/Tile*
N*,
_class"
 loc:@loss/dense_3_loss/truediv*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
8training/Adam/gradients/dense_3/Sigmoid_grad/SigmoidGradSigmoidGraddense_3/Sigmoidtraining/Adam/gradients/AddN*"
_class
loc:@dense_3/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
с
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/dense_3/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd*
T0*
_output_shapes
:

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul8training/Adam/gradients/dense_3/Sigmoid_grad/SigmoidGraddense_3/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_3/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Sigmoid8training/Adam/gradients/dense_3/Sigmoid_grad/SigmoidGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_3/MatMul*
T0*
_output_shapes

:
т
8training/Adam/gradients/dense_2/Sigmoid_grad/SigmoidGradSigmoidGraddense_2/Sigmoid2training/Adam/gradients/dense_3/MatMul_grad/MatMul*"
_class
loc:@dense_2/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
с
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/dense_2/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
T0*
_output_shapes
:

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul8training/Adam/gradients/dense_2/Sigmoid_grad/SigmoidGraddense_2/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_2/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Sigmoid8training/Adam/gradients/dense_2/Sigmoid_grad/SigmoidGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_2/MatMul*
T0*
_output_shapes

:
т
8training/Adam/gradients/dense_1/Sigmoid_grad/SigmoidGradSigmoidGraddense_1/Sigmoid2training/Adam/gradients/dense_2/MatMul_grad/MatMul*"
_class
loc:@dense_1/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
с
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/dense_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*"
_class
loc:@dense_1/BiasAdd*
T0*
_output_shapes
:

2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul8training/Adam/gradients/dense_1/Sigmoid_grad/SigmoidGraddense_1/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_1/MatMul*
T0*(
_output_shapes
:џџџџџџџџџ Z
ў
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape8training/Adam/gradients/dense_1/Sigmoid_grad/SigmoidGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes
:	 Z
Б
4training/Adam/gradients/flatten_1/Reshape_grad/ShapeShapemax_pooling2d_2/MaxPool*$
_class
loc:@flatten_1/Reshape*
T0*
_output_shapes
:*
out_type0

6training/Adam/gradients/flatten_1/Reshape_grad/ReshapeReshape2training/Adam/gradients/dense_1/MatMul_grad/MatMul4training/Adam/gradients/flatten_1/Reshape_grad/Shape*$
_class
loc:@flatten_1/Reshape*
T0*/
_output_shapes
:џџџџџџџџџ *
Tshape0
ш
@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Tanhmax_pooling2d_2/MaxPool6training/Adam/gradients/flatten_1/Reshape_grad/Reshape*
data_formatNHWC*
paddingVALID*
ksize
**
_class 
loc:@max_pooling2d_2/MaxPool*
T0*/
_output_shapes
:џџџџџџџџџ'' *
strides

ь
3training/Adam/gradients/conv2d_2/Tanh_grad/TanhGradTanhGradconv2d_2/Tanh@training/Adam/gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGrad* 
_class
loc:@conv2d_2/Tanh*
T0*/
_output_shapes
:џџџџџџџџџ'' 
о
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_2/Tanh_grad/TanhGrad*
data_formatNHWC*#
_class
loc:@conv2d_2/BiasAdd*
T0*
_output_shapes
: 
о
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNmax_pooling2d_1/MaxPoolconv2d_2/kernel/read*
N*'
_class
loc:@conv2d_2/convolution*
T0* 
_output_shapes
::*
out_type0
Г
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read3training/Adam/gradients/conv2d_2/Tanh_grad/TanhGrad*
data_formatNHWC*
paddingVALID*
	dilations
*'
_class
loc:@conv2d_2/convolution*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:џџџџџџџџџ.. *
strides

Б
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d_1/MaxPool:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_2/Tanh_grad/TanhGrad*
data_formatNHWC*
paddingVALID*
	dilations
*'
_class
loc:@conv2d_2/convolution*
T0*
use_cudnn_on_gpu(*&
_output_shapes
:  *
strides

ї
@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_1/Tanhmax_pooling2d_1/MaxPoolEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
data_formatNHWC*
paddingVALID*
ksize
**
_class 
loc:@max_pooling2d_1/MaxPool*
T0*/
_output_shapes
:џџџџџџџџџ]] *
strides

ь
3training/Adam/gradients/conv2d_1/Tanh_grad/TanhGradTanhGradconv2d_1/Tanh@training/Adam/gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGrad* 
_class
loc:@conv2d_1/Tanh*
T0*/
_output_shapes
:џџџџџџџџџ]] 
о
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/conv2d_1/Tanh_grad/TanhGrad*
data_formatNHWC*#
_class
loc:@conv2d_1/BiasAdd*
T0*
_output_shapes
: 
Ю
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNinput_1conv2d_1/kernel/read*
N*'
_class
loc:@conv2d_1/convolution*
T0* 
_output_shapes
::*
out_type0
Г
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read3training/Adam/gradients/conv2d_1/Tanh_grad/TanhGrad*
data_formatNHWC*
paddingVALID*
	dilations
*'
_class
loc:@conv2d_1/convolution*
T0*
use_cudnn_on_gpu(*/
_output_shapes
:џџџџџџџџџdd*
strides

Ё
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_1:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:13training/Adam/gradients/conv2d_1/Tanh_grad/TanhGrad*
data_formatNHWC*
paddingVALID*
	dilations
*'
_class
loc:@conv2d_1/convolution*
T0*
use_cudnn_on_gpu(*&
_output_shapes
: *
strides

_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ќ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *"
_class
loc:@Adam/iterations*
T0	*
_output_shapes
: 
`
training/Adam/CastCastAdam/iterations/read*

DstT0*

SrcT0	*
_output_shapes
: 
X
training/Adam/add/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_1Const*
_output_shapes
: *
valueB
 *  *
dtype0
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
|
#training/Adam/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"             *
dtype0
^
training/Adam/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*

index_type0*
T0*&
_output_shapes
: 

training/Adam/Variable
VariableV2*
	container *
shared_name *
shape: *&
_output_shapes
: *
dtype0
й
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*)
_class
loc:@training/Adam/Variable*
T0*&
_output_shapes
: *
validate_shape(

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
T0*&
_output_shapes
: 
b
training/Adam/zeros_1Const*
_output_shapes
: *
valueB *    *
dtype0

training/Adam/Variable_1
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
е
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
T0*
_output_shapes
: *
validate_shape(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
T0*
_output_shapes
: 
~
%training/Adam/zeros_2/shape_as_tensorConst*
_output_shapes
:*%
valueB"              *
dtype0
`
training/Adam/zeros_2/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Є
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*

index_type0*
T0*&
_output_shapes
:  

training/Adam/Variable_2
VariableV2*
	container *
shared_name *
shape:  *&
_output_shapes
:  *
dtype0
с
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
T0*&
_output_shapes
:  *
validate_shape(
Ё
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
T0*&
_output_shapes
:  
b
training/Adam/zeros_3Const*
_output_shapes
: *
valueB *    *
dtype0

training/Adam/Variable_3
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
е
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
T0*
_output_shapes
: *
validate_shape(

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
T0*
_output_shapes
: 
v
%training/Adam/zeros_4/shape_as_tensorConst*
_output_shapes
:*
valueB" -     *
dtype0
`
training/Adam/zeros_4/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*

index_type0*
T0*
_output_shapes
:	 Z

training/Adam/Variable_4
VariableV2*
	container *
shared_name *
shape:	 Z*
_output_shapes
:	 Z*
dtype0
к
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
T0*
_output_shapes
:	 Z*
validate_shape(

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
T0*
_output_shapes
:	 Z
b
training/Adam/zeros_5Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_5
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
е
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
T0*
_output_shapes
:
j
training/Adam/zeros_6Const*
_output_shapes

:*
valueB*    *
dtype0

training/Adam/Variable_6
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
й
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
T0*
_output_shapes

:*
validate_shape(

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
T0*
_output_shapes

:
b
training/Adam/zeros_7Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_7
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
е
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
T0*
_output_shapes
:
j
training/Adam/zeros_8Const*
_output_shapes

:*
valueB*    *
dtype0

training/Adam/Variable_8
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
й
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
T0*
_output_shapes

:*
validate_shape(

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
T0*
_output_shapes

:
b
training/Adam/zeros_9Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_9
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
е
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
T0*
_output_shapes
:

&training/Adam/zeros_10/shape_as_tensorConst*
_output_shapes
:*%
valueB"             *
dtype0
a
training/Adam/zeros_10/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ї
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*

index_type0*
T0*&
_output_shapes
: 

training/Adam/Variable_10
VariableV2*
	container *
shared_name *
shape: *&
_output_shapes
: *
dtype0
х
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
T0*&
_output_shapes
: *
validate_shape(
Є
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
T0*&
_output_shapes
: 
c
training/Adam/zeros_11Const*
_output_shapes
: *
valueB *    *
dtype0

training/Adam/Variable_11
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
й
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes
: *
validate_shape(

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes
: 

&training/Adam/zeros_12/shape_as_tensorConst*
_output_shapes
:*%
valueB"              *
dtype0
a
training/Adam/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ї
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*

index_type0*
T0*&
_output_shapes
:  

training/Adam/Variable_12
VariableV2*
	container *
shared_name *
shape:  *&
_output_shapes
:  *
dtype0
х
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
T0*&
_output_shapes
:  *
validate_shape(
Є
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
T0*&
_output_shapes
:  
c
training/Adam/zeros_13Const*
_output_shapes
: *
valueB *    *
dtype0

training/Adam/Variable_13
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0
й
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
T0*
_output_shapes
: *
validate_shape(

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
T0*
_output_shapes
: 
w
&training/Adam/zeros_14/shape_as_tensorConst*
_output_shapes
:*
valueB" -     *
dtype0
a
training/Adam/zeros_14/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
 
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*

index_type0*
T0*
_output_shapes
:	 Z

training/Adam/Variable_14
VariableV2*
	container *
shared_name *
shape:	 Z*
_output_shapes
:	 Z*
dtype0
о
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	 Z*
validate_shape(

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	 Z
c
training/Adam/zeros_15Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_15
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
T0*
_output_shapes
:
k
training/Adam/zeros_16Const*
_output_shapes

:*
valueB*    *
dtype0

training/Adam/Variable_16
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
н
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*,
_class"
 loc:@training/Adam/Variable_16*
T0*
_output_shapes

:*
validate_shape(

training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
T0*
_output_shapes

:
c
training/Adam/zeros_17Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_17
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*,
_class"
 loc:@training/Adam/Variable_17*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
T0*
_output_shapes
:
k
training/Adam/zeros_18Const*
_output_shapes

:*
valueB*    *
dtype0

training/Adam/Variable_18
VariableV2*
	container *
shared_name *
shape
:*
_output_shapes

:*
dtype0
н
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*,
_class"
 loc:@training/Adam/Variable_18*
T0*
_output_shapes

:*
validate_shape(

training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
T0*
_output_shapes

:
c
training/Adam/zeros_19Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_19
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*,
_class"
 loc:@training/Adam/Variable_19*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
T0*
_output_shapes
:
p
&training/Adam/zeros_20/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_20/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_20
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*,
_class"
 loc:@training/Adam/Variable_20*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
T0*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_21/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_21
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*,
_class"
 loc:@training/Adam/Variable_21*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
T0*
_output_shapes
:
p
&training/Adam/zeros_22/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_22/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_22
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*,
_class"
 loc:@training/Adam/Variable_22*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
T0*
_output_shapes
:
p
&training/Adam/zeros_23/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_23/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_23
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*,
_class"
 loc:@training/Adam/Variable_23*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
T0*
_output_shapes
:
p
&training/Adam/zeros_24/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_24/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_24Fill&training/Adam/zeros_24/shape_as_tensortraining/Adam/zeros_24/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_24
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*,
_class"
 loc:@training/Adam/Variable_24*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
T0*
_output_shapes
:
p
&training/Adam/zeros_25/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_25/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_25Fill&training/Adam/zeros_25/shape_as_tensortraining/Adam/zeros_25/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_25
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*,
_class"
 loc:@training/Adam/Variable_25*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
T0*
_output_shapes
:
p
&training/Adam/zeros_26/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_26/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_26
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
use_locking(*,
_class"
 loc:@training/Adam/Variable_26*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
T0*
_output_shapes
:
p
&training/Adam/zeros_27/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_27/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_27Fill&training/Adam/zeros_27/shape_as_tensortraining/Adam/zeros_27/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_27
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
use_locking(*,
_class"
 loc:@training/Adam/Variable_27*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
T0*
_output_shapes
:
p
&training/Adam/zeros_28/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_28/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_28
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
use_locking(*,
_class"
 loc:@training/Adam/Variable_28*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
T0*
_output_shapes
:
p
&training/Adam/zeros_29/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_29/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

training/Adam/zeros_29Fill&training/Adam/zeros_29/shape_as_tensortraining/Adam/zeros_29/Const*

index_type0*
T0*
_output_shapes
:

training/Adam/Variable_29
VariableV2*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
й
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*,
_class"
 loc:@training/Adam/Variable_29*
T0*
_output_shapes
:*
validate_shape(

training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
T0*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ј
training/Adam/mul_2Multraining/Adam/sub_2Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*&
_output_shapes
: 
}
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/SquareSquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
: 
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
: 
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
: 
Z
training/Adam/Const_2Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_3Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*&
_output_shapes
: 

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
: 
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
: 
Z
training/Adam/add_3/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
: 
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*&
_output_shapes
: 
z
training/Adam/sub_4Subconv2d_1/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
: 
а
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*)
_class
loc:@training/Adam/Variable*
T0*&
_output_shapes
: *
validate_shape(
и
training/Adam/Assign_1Assigntraining/Adam/Variable_10training/Adam/add_2*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
T0*&
_output_shapes
: *
validate_shape(
Ф
training/Adam/Assign_2Assignconv2d_1/kerneltraining/Adam/sub_4*
use_locking(*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: *
validate_shape(
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_59training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
: 
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
: 
Z
training/Adam/sub_6/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
: 
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
: 
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_5Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
: 

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
: 
Z
training/Adam/add_6/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
: 
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
: 
l
training/Adam/sub_7Subconv2d_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
: 
Ъ
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
T0*
_output_shapes
: *
validate_shape(
Ь
training/Adam/Assign_4Assigntraining/Adam/Variable_11training/Adam/add_5*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes
: *
validate_shape(
Д
training/Adam/Assign_5Assignconv2d_1/biastraining/Adam/sub_7*
use_locking(* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
: *
validate_shape(
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*&
_output_shapes
:  
Z
training/Adam/sub_8/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
Љ
training/Adam/mul_12Multraining/Adam/sub_8Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
:  
~
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*&
_output_shapes
:  
Z
training/Adam/sub_9/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:  
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*&
_output_shapes
:  
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*&
_output_shapes
:  
Z
training/Adam/Const_6Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_7Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*&
_output_shapes
:  

training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
:  
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
:  
Z
training/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
:  
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*&
_output_shapes
:  
{
training/Adam/sub_10Subconv2d_2/kernel/readtraining/Adam/truediv_3*
T0*&
_output_shapes
:  
ж
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
T0*&
_output_shapes
:  *
validate_shape(
и
training/Adam/Assign_7Assigntraining/Adam/Variable_12training/Adam/add_8*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
T0*&
_output_shapes
:  *
validate_shape(
Х
training/Adam/Assign_8Assignconv2d_2/kerneltraining/Adam/sub_10*
use_locking(*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:  *
validate_shape(
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_119training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
: 
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
: 
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
: 
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_9Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
: 

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
: 
[
training/Adam/add_12/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
: 
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
: 
m
training/Adam/sub_13Subconv2d_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
: 
Ы
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
T0*
_output_shapes
: *
validate_shape(
Ю
training/Adam/Assign_10Assigntraining/Adam/Variable_13training/Adam/add_11*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
T0*
_output_shapes
: *
validate_shape(
Ж
training/Adam/Assign_11Assignconv2d_2/biastraining/Adam/sub_13*
use_locking(* 
_class
loc:@conv2d_2/bias*
T0*
_output_shapes
: *
validate_shape(
v
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes
:	 Z
[
training/Adam/sub_14/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	 Z
q
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes
:	 Z
w
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes
:	 Z
[
training/Adam/sub_15/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	 Z
s
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes
:	 Z
q
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes
:	 Z
n
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes
:	 Z
[
training/Adam/Const_10Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_11Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes
:	 Z

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes
:	 Z
e
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes
:	 Z
[
training/Adam/add_15/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
s
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes
:	 Z
x
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes
:	 Z
s
training/Adam/sub_16Subdense_1/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes
:	 Z
б
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
T0*
_output_shapes
:	 Z*
validate_shape(
г
training/Adam/Assign_13Assigntraining/Adam/Variable_14training/Adam/add_14*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	 Z*
validate_shape(
Н
training/Adam/Assign_14Assigndense_1/kerneltraining/Adam/sub_16*
use_locking(*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	 Z*
validate_shape(
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
:
[
training/Adam/sub_17/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_13Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
[
training/Adam/add_18/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:
l
training/Adam/sub_19Subdense_1/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
Ь
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
T0*
_output_shapes
:*
validate_shape(
Ю
training/Adam/Assign_16Assigntraining/Adam/Variable_15training/Adam/add_17*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
T0*
_output_shapes
:*
validate_shape(
Д
training/Adam/Assign_17Assigndense_1/biastraining/Adam/sub_19*
use_locking(*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:*
validate_shape(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:
[
training/Adam/sub_20/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_16/read*
T0*
_output_shapes

:
[
training/Adam/sub_21/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:
[
training/Adam/Const_14Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_15Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*
_output_shapes

:

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:
[
training/Adam/add_21/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

:
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:
r
training/Adam/sub_22Subdense_2/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes

:
а
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
T0*
_output_shapes

:*
validate_shape(
в
training/Adam/Assign_19Assigntraining/Adam/Variable_16training/Adam/add_20*
use_locking(*,
_class"
 loc:@training/Adam/Variable_16*
T0*
_output_shapes

:*
validate_shape(
М
training/Adam/Assign_20Assigndense_2/kerneltraining/Adam/sub_22*
use_locking(*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:
[
training/Adam/sub_23/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_17/read*
T0*
_output_shapes
:
[
training/Adam/sub_24/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:
[
training/Adam/Const_16Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_17Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes
:
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:
[
training/Adam/add_24/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:
l
training/Adam/sub_25Subdense_2/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:
Ь
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
T0*
_output_shapes
:*
validate_shape(
Ю
training/Adam/Assign_22Assigntraining/Adam/Variable_17training/Adam/add_23*
use_locking(*,
_class"
 loc:@training/Adam/Variable_17*
T0*
_output_shapes
:*
validate_shape(
Д
training/Adam/Assign_23Assigndense_2/biastraining/Adam/sub_25*
use_locking(*
_class
loc:@dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
u
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:
[
training/Adam/sub_26/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_42Multraining/Adam/sub_264training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
p
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*
T0*
_output_shapes

:
v
training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_18/read*
T0*
_output_shapes

:
[
training/Adam/sub_27/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_8Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
r
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*
_output_shapes

:
p
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*
_output_shapes

:
m
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*
_output_shapes

:
[
training/Adam/Const_18Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_19Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*
_output_shapes

:

training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*
_output_shapes

:
d
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*
_output_shapes

:
[
training/Adam/add_27/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
r
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*
_output_shapes

:
w
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*
_output_shapes

:
r
training/Adam/sub_28Subdense_3/kernel/readtraining/Adam/truediv_9*
T0*
_output_shapes

:
а
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
T0*
_output_shapes

:*
validate_shape(
в
training/Adam/Assign_25Assigntraining/Adam/Variable_18training/Adam/add_26*
use_locking(*,
_class"
 loc:@training/Adam/Variable_18*
T0*
_output_shapes

:*
validate_shape(
М
training/Adam/Assign_26Assigndense_3/kerneltraining/Adam/sub_28*
use_locking(*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes

:*
validate_shape(
q
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:
[
training/Adam/sub_29/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_47Multraining/Adam/sub_298training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes
:
r
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_19/read*
T0*
_output_shapes
:
[
training/Adam/sub_30/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_9Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes
:
l
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes
:
i
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes
:
[
training/Adam/Const_20Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_21Const*
_output_shapes
: *
valueB
 *  *
dtype0

&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
T0*
_output_shapes
:

training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
T0*
_output_shapes
:
b
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
T0*
_output_shapes
:
[
training/Adam/add_30/yConst*
_output_shapes
: *
valueB
 *Пж3*
dtype0
o
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes
:
t
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
T0*
_output_shapes
:
m
training/Adam/sub_31Subdense_3/bias/readtraining/Adam/truediv_10*
T0*
_output_shapes
:
Ь
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
T0*
_output_shapes
:*
validate_shape(
Ю
training/Adam/Assign_28Assigntraining/Adam/Variable_19training/Adam/add_29*
use_locking(*,
_class"
 loc:@training/Adam/Variable_19*
T0*
_output_shapes
:*
validate_shape(
Д
training/Adam/Assign_29Assigndense_3/biastraining/Adam/sub_31*
use_locking(*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:*
validate_shape(
г
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9

IsVariableInitializedIsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_1IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_2IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_3IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_4IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_5IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_6IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_7IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_8IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_9IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_10IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
dtype0	
{
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
_output_shapes
: *
dtype0

IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
_output_shapes
: *
dtype0

IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
dtype0

IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
dtype0

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes
: *
dtype0

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: *
dtype0

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes
: *
dtype0

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: *
dtype0

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes
: *
dtype0

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: *
dtype0

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes
: *
dtype0

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
: *
dtype0

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: *
dtype0

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
: *
dtype0

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes
: *
dtype0

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: *
dtype0

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes
: *
dtype0

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
: *
dtype0

IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
: *
dtype0

IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
: *
dtype0

IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
: *
dtype0

IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
: *
dtype0

IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
: *
dtype0

IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
: *
dtype0

IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
: *
dtype0

IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
: *
dtype0

IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
: *
dtype0

IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
: *
dtype0

IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
_output_shapes
: *
dtype0

IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
: *
dtype0

IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
_output_shapes
: *
dtype0

IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_27*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
: *
dtype0

IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
_output_shapes
: *
dtype0

IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes
: *
dtype0
ю

init_2NoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign
d
PlaceholderPlaceholder*
shape:dd*"
_output_shapes
:dd*
dtype0
p
Placeholder_1Placeholder*
shape:џџџџџџџџџ*'
_output_shapes
:џџџџџџџџџ*
dtype0
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_abdc31c7e48a4492991241c1a4257216/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
щ
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
valueB-BAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_4Btraining/Adam/Variable_5Btraining/Adam/Variable_6Btraining/Adam/Variable_7Btraining/Adam/Variable_8Btraining/Adam/Variable_9*
dtype0
Ь
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ї	
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAdam/beta_1Adam/beta_2
Adam/decayAdam/iterationsAdam/lrconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kerneltraining/Adam/Variabletraining/Adam/Variable_1training/Adam/Variable_10training/Adam/Variable_11training/Adam/Variable_12training/Adam/Variable_13training/Adam/Variable_14training/Adam/Variable_15training/Adam/Variable_16training/Adam/Variable_17training/Adam/Variable_18training/Adam/Variable_19training/Adam/Variable_2training/Adam/Variable_20training/Adam/Variable_21training/Adam/Variable_22training/Adam/Variable_23training/Adam/Variable_24training/Adam/Variable_25training/Adam/Variable_26training/Adam/Variable_27training/Adam/Variable_28training/Adam/Variable_29training/Adam/Variable_3training/Adam/Variable_4training/Adam/Variable_5training/Adam/Variable_6training/Adam/Variable_7training/Adam/Variable_8training/Adam/Variable_9"/device:CPU:0*;
dtypes1
/2-	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
ь
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
valueB-BAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_4Btraining/Adam/Variable_5Btraining/Adam/Variable_6Btraining/Adam/Variable_7Btraining/Adam/Variable_8Btraining/Adam/Variable_9*
dtype0
Я
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ў
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*;
dtypes1
/2-	*Ъ
_output_shapesЗ
Д:::::::::::::::::::::::::::::::::::::::::::::

save/AssignAssignAdam/beta_1save/RestoreV2*
use_locking(*
_class
loc:@Adam/beta_1*
T0*
_output_shapes
: *
validate_shape(
 
save/Assign_1AssignAdam/beta_2save/RestoreV2:1*
use_locking(*
_class
loc:@Adam/beta_2*
T0*
_output_shapes
: *
validate_shape(

save/Assign_2Assign
Adam/decaysave/RestoreV2:2*
use_locking(*
_class
loc:@Adam/decay*
T0*
_output_shapes
: *
validate_shape(
Ј
save/Assign_3AssignAdam/iterationssave/RestoreV2:3*
use_locking(*"
_class
loc:@Adam/iterations*
T0	*
_output_shapes
: *
validate_shape(

save/Assign_4AssignAdam/lrsave/RestoreV2:4*
use_locking(*
_class
loc:@Adam/lr*
T0*
_output_shapes
: *
validate_shape(
Ј
save/Assign_5Assignconv2d_1/biassave/RestoreV2:5*
use_locking(* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
: *
validate_shape(
И
save/Assign_6Assignconv2d_1/kernelsave/RestoreV2:6*
use_locking(*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: *
validate_shape(
Ј
save/Assign_7Assignconv2d_2/biassave/RestoreV2:7*
use_locking(* 
_class
loc:@conv2d_2/bias*
T0*
_output_shapes
: *
validate_shape(
И
save/Assign_8Assignconv2d_2/kernelsave/RestoreV2:8*
use_locking(*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:  *
validate_shape(
І
save/Assign_9Assigndense_1/biassave/RestoreV2:9*
use_locking(*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:*
validate_shape(
Б
save/Assign_10Assigndense_1/kernelsave/RestoreV2:10*
use_locking(*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	 Z*
validate_shape(
Ј
save/Assign_11Assigndense_2/biassave/RestoreV2:11*
use_locking(*
_class
loc:@dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
А
save/Assign_12Assigndense_2/kernelsave/RestoreV2:12*
use_locking(*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(
Ј
save/Assign_13Assigndense_3/biassave/RestoreV2:13*
use_locking(*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:*
validate_shape(
А
save/Assign_14Assigndense_3/kernelsave/RestoreV2:14*
use_locking(*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes

:*
validate_shape(
Ш
save/Assign_15Assigntraining/Adam/Variablesave/RestoreV2:15*
use_locking(*)
_class
loc:@training/Adam/Variable*
T0*&
_output_shapes
: *
validate_shape(
Р
save/Assign_16Assigntraining/Adam/Variable_1save/RestoreV2:16*
use_locking(*+
_class!
loc:@training/Adam/Variable_1*
T0*
_output_shapes
: *
validate_shape(
Ю
save/Assign_17Assigntraining/Adam/Variable_10save/RestoreV2:17*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10*
T0*&
_output_shapes
: *
validate_shape(
Т
save/Assign_18Assigntraining/Adam/Variable_11save/RestoreV2:18*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes
: *
validate_shape(
Ю
save/Assign_19Assigntraining/Adam/Variable_12save/RestoreV2:19*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12*
T0*&
_output_shapes
:  *
validate_shape(
Т
save/Assign_20Assigntraining/Adam/Variable_13save/RestoreV2:20*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
T0*
_output_shapes
: *
validate_shape(
Ч
save/Assign_21Assigntraining/Adam/Variable_14save/RestoreV2:21*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	 Z*
validate_shape(
Т
save/Assign_22Assigntraining/Adam/Variable_15save/RestoreV2:22*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
T0*
_output_shapes
:*
validate_shape(
Ц
save/Assign_23Assigntraining/Adam/Variable_16save/RestoreV2:23*
use_locking(*,
_class"
 loc:@training/Adam/Variable_16*
T0*
_output_shapes

:*
validate_shape(
Т
save/Assign_24Assigntraining/Adam/Variable_17save/RestoreV2:24*
use_locking(*,
_class"
 loc:@training/Adam/Variable_17*
T0*
_output_shapes
:*
validate_shape(
Ц
save/Assign_25Assigntraining/Adam/Variable_18save/RestoreV2:25*
use_locking(*,
_class"
 loc:@training/Adam/Variable_18*
T0*
_output_shapes

:*
validate_shape(
Т
save/Assign_26Assigntraining/Adam/Variable_19save/RestoreV2:26*
use_locking(*,
_class"
 loc:@training/Adam/Variable_19*
T0*
_output_shapes
:*
validate_shape(
Ь
save/Assign_27Assigntraining/Adam/Variable_2save/RestoreV2:27*
use_locking(*+
_class!
loc:@training/Adam/Variable_2*
T0*&
_output_shapes
:  *
validate_shape(
Т
save/Assign_28Assigntraining/Adam/Variable_20save/RestoreV2:28*
use_locking(*,
_class"
 loc:@training/Adam/Variable_20*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_29Assigntraining/Adam/Variable_21save/RestoreV2:29*
use_locking(*,
_class"
 loc:@training/Adam/Variable_21*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_30Assigntraining/Adam/Variable_22save/RestoreV2:30*
use_locking(*,
_class"
 loc:@training/Adam/Variable_22*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_31Assigntraining/Adam/Variable_23save/RestoreV2:31*
use_locking(*,
_class"
 loc:@training/Adam/Variable_23*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_32Assigntraining/Adam/Variable_24save/RestoreV2:32*
use_locking(*,
_class"
 loc:@training/Adam/Variable_24*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_33Assigntraining/Adam/Variable_25save/RestoreV2:33*
use_locking(*,
_class"
 loc:@training/Adam/Variable_25*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_34Assigntraining/Adam/Variable_26save/RestoreV2:34*
use_locking(*,
_class"
 loc:@training/Adam/Variable_26*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_35Assigntraining/Adam/Variable_27save/RestoreV2:35*
use_locking(*,
_class"
 loc:@training/Adam/Variable_27*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_36Assigntraining/Adam/Variable_28save/RestoreV2:36*
use_locking(*,
_class"
 loc:@training/Adam/Variable_28*
T0*
_output_shapes
:*
validate_shape(
Т
save/Assign_37Assigntraining/Adam/Variable_29save/RestoreV2:37*
use_locking(*,
_class"
 loc:@training/Adam/Variable_29*
T0*
_output_shapes
:*
validate_shape(
Р
save/Assign_38Assigntraining/Adam/Variable_3save/RestoreV2:38*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
T0*
_output_shapes
: *
validate_shape(
Х
save/Assign_39Assigntraining/Adam/Variable_4save/RestoreV2:39*
use_locking(*+
_class!
loc:@training/Adam/Variable_4*
T0*
_output_shapes
:	 Z*
validate_shape(
Р
save/Assign_40Assigntraining/Adam/Variable_5save/RestoreV2:40*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
T0*
_output_shapes
:*
validate_shape(
Ф
save/Assign_41Assigntraining/Adam/Variable_6save/RestoreV2:41*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
T0*
_output_shapes

:*
validate_shape(
Р
save/Assign_42Assigntraining/Adam/Variable_7save/RestoreV2:42*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
T0*
_output_shapes
:*
validate_shape(
Ф
save/Assign_43Assigntraining/Adam/Variable_8save/RestoreV2:43*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
T0*
_output_shapes

:*
validate_shape(
Р
save/Assign_44Assigntraining/Adam/Variable_9save/RestoreV2:44*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
T0*
_output_shapes
:*
validate_shape(

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"ш'
	variablesк'з'
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08"ђ'
trainable_variablesк'з'
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08*
predict_imagesy
)
images
Placeholder:0dd0
scores&
Placeholder_1:0џџџџџџџџџtensorflow/serving/predict
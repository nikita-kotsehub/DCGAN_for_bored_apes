??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
d??*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes

:??*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes

:??*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes

:??*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes

:??*
dtype0
?
conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameconv2d_transpose_12/kernel
?
.conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameconv2d_transpose_13/kernel
?
.conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_14/gamma
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_14/beta
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_14/moving_mean
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_14/moving_variance
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_14/kernel
?
.conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/kernel*'
_output_shapes
:@?*
dtype0
?
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_15/moving_mean
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_15/moving_variance
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_15/kernel
?
.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
regularization_losses
trainable_variables
	variables
	keras_api

signatures
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
^

*kernel
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
^

<kernel
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
^

Nkernel
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
^

`kernel
aregularization_losses
btrainable_variables
c	variables
d	keras_api
 
^
0
1
2
*3
04
15
<6
B7
C8
N9
T10
U11
`12
?
0
1
2
3
4
*5
06
17
28
39
<10
B11
C12
D13
E14
N15
T16
U17
V18
W19
`20
?
enon_trainable_variables
regularization_losses

flayers
glayer_regularization_losses
hlayer_metrics
imetrics
trainable_variables
	variables
 
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
jnon_trainable_variables
regularization_losses

klayers
llayer_regularization_losses
mlayer_metrics
nmetrics
trainable_variables
	variables
 
ge
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
?
onon_trainable_variables
regularization_losses

players
qlayer_regularization_losses
rlayer_metrics
smetrics
trainable_variables
 	variables
 
 
 
?
tnon_trainable_variables
"regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
xmetrics
#trainable_variables
$	variables
 
 
 
?
ynon_trainable_variables
&regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
}metrics
'trainable_variables
(	variables
fd
VARIABLE_VALUEconv2d_transpose_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0
?
~non_trainable_variables
+regularization_losses

layers
 ?layer_regularization_losses
?layer_metrics
?metrics
,trainable_variables
-	variables
 
ge
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
22
33
?
?non_trainable_variables
4regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
5trainable_variables
6	variables
 
 
 
?
?non_trainable_variables
8regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
9trainable_variables
:	variables
fd
VARIABLE_VALUEconv2d_transpose_13/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

<0

<0
?
?non_trainable_variables
=regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
>trainable_variables
?	variables
 
ge
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
D2
E3
?
?non_trainable_variables
Fregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Gtrainable_variables
H	variables
 
 
 
?
?non_trainable_variables
Jregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ktrainable_variables
L	variables
fd
VARIABLE_VALUEconv2d_transpose_14/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

N0

N0
?
?non_trainable_variables
Oregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ptrainable_variables
Q	variables
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
V2
W3
?
?non_trainable_variables
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ytrainable_variables
Z	variables
 
 
 
?
?non_trainable_variables
\regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
]trainable_variables
^	variables
fd
VARIABLE_VALUEconv2d_transpose_15/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

`0

`0
?
?non_trainable_variables
aregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
btrainable_variables
c	variables
8
0
1
22
33
D4
E5
V6
W7
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

V0
W1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_dense_7_inputPlaceholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_7_inputdense_7/kernel&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betaconv2d_transpose_12/kernelbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_transpose_13/kernelbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_transpose_14/kernelbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_transpose_15/kernel*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_25224
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp.conv2d_transpose_12/kernel/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp.conv2d_transpose_13/kernel/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp.conv2d_transpose_14/kernel/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp.conv2d_transpose_15/kernel/Read/ReadVariableOpConst*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_26045
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_transpose_12/kernelbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_transpose_13/kernelbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_varianceconv2d_transpose_14/kernelbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceconv2d_transpose_15/kernel*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_26118??
?
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_25738

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_13_layer_call_fn_25805

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_242302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_24816
dense_7_input
unknown:
d??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
	unknown_3:
??%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?%

unknown_14:@?

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_247712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25368

inputs:
&dense_7_matmul_readvariableop_resource:
d??H
8batch_normalization_12_batchnorm_readvariableop_resource:
??L
<batch_normalization_12_batchnorm_mul_readvariableop_resource:
??J
:batch_normalization_12_batchnorm_readvariableop_1_resource:
??J
:batch_normalization_12_batchnorm_readvariableop_2_resource:
??X
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:??=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?X
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource:??=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource:@?<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:@
identity??/batch_normalization_12/batchnorm/ReadVariableOp?1batch_normalization_12/batchnorm/ReadVariableOp_1?1batch_normalization_12/batchnorm/ReadVariableOp_2?3batch_normalization_12/batchnorm/mul/ReadVariableOp?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_7/MatMul?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp?
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_12/batchnorm/add/y?
$batch_normalization_12/batchnorm/addAddV27batch_normalization_12/batchnorm/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes

:??2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Muldense_7/MatMul:product:0(batch_normalization_12/batchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2(
&batch_normalization_12/batchnorm/mul_1?
1batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes

:??*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_1?
&batch_normalization_12/batchnorm/mul_2Mul9batch_normalization_12/batchnorm/ReadVariableOp_1:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes

:??2(
&batch_normalization_12/batchnorm/mul_2?
1batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes

:??*
dtype023
1batch_normalization_12/batchnorm/ReadVariableOp_2?
$batch_normalization_12/batchnorm/subSub9batch_normalization_12/batchnorm/ReadVariableOp_2:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2(
&batch_normalization_12/batchnorm/add_1?
leaky_re_lu_27/LeakyRelu	LeakyRelu*batch_normalization_12/batchnorm/add_1:z:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_27/LeakyRelux
reshape_3/ShapeShape&leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2y
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_3/Reshape/shape/3?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshape&leaky_re_lu_27/LeakyRelu:activations:0 reshape_3/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_3/Reshape?
conv2d_transpose_12/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2}
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_12/conv2d_transpose:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3?
leaky_re_lu_28/LeakyRelu	LeakyRelu+batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_28/LeakyRelu?
conv2d_transpose_13/ShapeShape&leaky_re_lu_28/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2}
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_28/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_13/conv2d_transpose:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3?
leaky_re_lu_29/LeakyRelu	LeakyRelu+batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_29/LeakyRelu?
conv2d_transpose_14/ShapeShape&leaky_re_lu_29/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_29/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_14/conv2d_transpose:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
leaky_re_lu_30/LeakyRelu	LeakyRelu+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_re_lu_30/LeakyRelu?
conv2d_transpose_15/ShapeShape&leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
conv2d_transpose_15/TanhTanh-conv2d_transpose_15/conv2d_transpose:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_15/Tanh?	
IdentityIdentityconv2d_transpose_15/Tanh:y:00^batch_normalization_12/batchnorm/ReadVariableOp2^batch_normalization_12/batchnorm/ReadVariableOp_12^batch_normalization_12/batchnorm/ReadVariableOp_24^batch_normalization_12/batchnorm/mul/ReadVariableOp7^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_14^conv2d_transpose_12/conv2d_transpose/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2f
1batch_normalization_12/batchnorm/ReadVariableOp_11batch_normalization_12/batchnorm/ReadVariableOp_12f
1batch_normalization_12/batchnorm/ReadVariableOp_21batch_normalization_12/batchnorm/ReadVariableOp_22j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_23963
dense_7_inputG
3sequential_5_dense_7_matmul_readvariableop_resource:
d??U
Esequential_5_batch_normalization_12_batchnorm_readvariableop_resource:
??Y
Isequential_5_batch_normalization_12_batchnorm_mul_readvariableop_resource:
??W
Gsequential_5_batch_normalization_12_batchnorm_readvariableop_1_resource:
??W
Gsequential_5_batch_normalization_12_batchnorm_readvariableop_2_resource:
??e
Isequential_5_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:??J
;sequential_5_batch_normalization_13_readvariableop_resource:	?L
=sequential_5_batch_normalization_13_readvariableop_1_resource:	?[
Lsequential_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?e
Isequential_5_conv2d_transpose_13_conv2d_transpose_readvariableop_resource:??J
;sequential_5_batch_normalization_14_readvariableop_resource:	?L
=sequential_5_batch_normalization_14_readvariableop_1_resource:	?[
Lsequential_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?d
Isequential_5_conv2d_transpose_14_conv2d_transpose_readvariableop_resource:@?I
;sequential_5_batch_normalization_15_readvariableop_resource:@K
=sequential_5_batch_normalization_15_readvariableop_1_resource:@Z
Lsequential_5_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_5_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@c
Isequential_5_conv2d_transpose_15_conv2d_transpose_readvariableop_resource:@
identity??<sequential_5/batch_normalization_12/batchnorm/ReadVariableOp?>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1?>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2?@sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOp?Csequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_13/ReadVariableOp?4sequential_5/batch_normalization_13/ReadVariableOp_1?Csequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_14/ReadVariableOp?4sequential_5/batch_normalization_14/ReadVariableOp_1?Csequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_15/ReadVariableOp?4sequential_5/batch_normalization_15/ReadVariableOp_1?@sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?@sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?@sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?@sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?*sequential_5/dense_7/MatMul/ReadVariableOp?
*sequential_5/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02,
*sequential_5/dense_7/MatMul/ReadVariableOp?
sequential_5/dense_7/MatMulMatMuldense_7_input2sequential_5/dense_7/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
sequential_5/dense_7/MatMul?
<sequential_5/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpEsequential_5_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype02>
<sequential_5/batch_normalization_12/batchnorm/ReadVariableOp?
3sequential_5/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_5/batch_normalization_12/batchnorm/add/y?
1sequential_5/batch_normalization_12/batchnorm/addAddV2Dsequential_5/batch_normalization_12/batchnorm/ReadVariableOp:value:0<sequential_5/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes

:??23
1sequential_5/batch_normalization_12/batchnorm/add?
3sequential_5/batch_normalization_12/batchnorm/RsqrtRsqrt5sequential_5/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes

:??25
3sequential_5/batch_normalization_12/batchnorm/Rsqrt?
@sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_5_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype02B
@sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOp?
1sequential_5/batch_normalization_12/batchnorm/mulMul7sequential_5/batch_normalization_12/batchnorm/Rsqrt:y:0Hsequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??23
1sequential_5/batch_normalization_12/batchnorm/mul?
3sequential_5/batch_normalization_12/batchnorm/mul_1Mul%sequential_5/dense_7/MatMul:product:05sequential_5/batch_normalization_12/batchnorm/mul:z:0*
T0*)
_output_shapes
:???????????25
3sequential_5/batch_normalization_12/batchnorm/mul_1?
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_5_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes

:??*
dtype02@
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1?
3sequential_5/batch_normalization_12/batchnorm/mul_2MulFsequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1:value:05sequential_5/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes

:??25
3sequential_5/batch_normalization_12/batchnorm/mul_2?
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_5_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes

:??*
dtype02@
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2?
1sequential_5/batch_normalization_12/batchnorm/subSubFsequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2:value:07sequential_5/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes

:??23
1sequential_5/batch_normalization_12/batchnorm/sub?
3sequential_5/batch_normalization_12/batchnorm/add_1AddV27sequential_5/batch_normalization_12/batchnorm/mul_1:z:05sequential_5/batch_normalization_12/batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????25
3sequential_5/batch_normalization_12/batchnorm/add_1?
%sequential_5/leaky_re_lu_27/LeakyRelu	LeakyRelu7sequential_5/batch_normalization_12/batchnorm/add_1:z:0*)
_output_shapes
:???????????*
alpha%???>2'
%sequential_5/leaky_re_lu_27/LeakyRelu?
sequential_5/reshape_3/ShapeShape3sequential_5/leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential_5/reshape_3/Shape?
*sequential_5/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/reshape_3/strided_slice/stack?
,sequential_5/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_3/strided_slice/stack_1?
,sequential_5/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/reshape_3/strided_slice/stack_2?
$sequential_5/reshape_3/strided_sliceStridedSlice%sequential_5/reshape_3/Shape:output:03sequential_5/reshape_3/strided_slice/stack:output:05sequential_5/reshape_3/strided_slice/stack_1:output:05sequential_5/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/reshape_3/strided_slice?
&sequential_5/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_3/Reshape/shape/1?
&sequential_5/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_5/reshape_3/Reshape/shape/2?
&sequential_5/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential_5/reshape_3/Reshape/shape/3?
$sequential_5/reshape_3/Reshape/shapePack-sequential_5/reshape_3/strided_slice:output:0/sequential_5/reshape_3/Reshape/shape/1:output:0/sequential_5/reshape_3/Reshape/shape/2:output:0/sequential_5/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$sequential_5/reshape_3/Reshape/shape?
sequential_5/reshape_3/ReshapeReshape3sequential_5/leaky_re_lu_27/LeakyRelu:activations:0-sequential_5/reshape_3/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2 
sequential_5/reshape_3/Reshape?
&sequential_5/conv2d_transpose_12/ShapeShape'sequential_5/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_12/Shape?
4sequential_5/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_5/conv2d_transpose_12/strided_slice/stack?
6sequential_5/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_12/strided_slice/stack_1?
6sequential_5/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_12/strided_slice/stack_2?
.sequential_5/conv2d_transpose_12/strided_sliceStridedSlice/sequential_5/conv2d_transpose_12/Shape:output:0=sequential_5/conv2d_transpose_12/strided_slice/stack:output:0?sequential_5/conv2d_transpose_12/strided_slice/stack_1:output:0?sequential_5/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_5/conv2d_transpose_12/strided_slice?
(sequential_5/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/conv2d_transpose_12/stack/1?
(sequential_5/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/conv2d_transpose_12/stack/2?
(sequential_5/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_5/conv2d_transpose_12/stack/3?
&sequential_5/conv2d_transpose_12/stackPack7sequential_5/conv2d_transpose_12/strided_slice:output:01sequential_5/conv2d_transpose_12/stack/1:output:01sequential_5/conv2d_transpose_12/stack/2:output:01sequential_5/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_12/stack?
6sequential_5/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_5/conv2d_transpose_12/strided_slice_1/stack?
8sequential_5/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_12/strided_slice_1/stack_1?
8sequential_5/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_12/strided_slice_1/stack_2?
0sequential_5/conv2d_transpose_12/strided_slice_1StridedSlice/sequential_5/conv2d_transpose_12/stack:output:0?sequential_5/conv2d_transpose_12/strided_slice_1/stack:output:0Asequential_5/conv2d_transpose_12/strided_slice_1/stack_1:output:0Asequential_5/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_5/conv2d_transpose_12/strided_slice_1?
@sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_5_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
1sequential_5/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput/sequential_5/conv2d_transpose_12/stack:output:0Hsequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0'sequential_5/reshape_3/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_5/conv2d_transpose_12/conv2d_transpose?
2sequential_5/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_5/batch_normalization_13/ReadVariableOp?
4sequential_5/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_5/batch_normalization_13/ReadVariableOp_1?
Csequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
Esequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
4sequential_5/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3:sequential_5/conv2d_transpose_12/conv2d_transpose:output:0:sequential_5/batch_normalization_13/ReadVariableOp:value:0<sequential_5/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_5/batch_normalization_13/FusedBatchNormV3?
%sequential_5/leaky_re_lu_28/LeakyRelu	LeakyRelu8sequential_5/batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2'
%sequential_5/leaky_re_lu_28/LeakyRelu?
&sequential_5/conv2d_transpose_13/ShapeShape3sequential_5/leaky_re_lu_28/LeakyRelu:activations:0*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_13/Shape?
4sequential_5/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_5/conv2d_transpose_13/strided_slice/stack?
6sequential_5/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_13/strided_slice/stack_1?
6sequential_5/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_13/strided_slice/stack_2?
.sequential_5/conv2d_transpose_13/strided_sliceStridedSlice/sequential_5/conv2d_transpose_13/Shape:output:0=sequential_5/conv2d_transpose_13/strided_slice/stack:output:0?sequential_5/conv2d_transpose_13/strided_slice/stack_1:output:0?sequential_5/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_5/conv2d_transpose_13/strided_slice?
(sequential_5/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/conv2d_transpose_13/stack/1?
(sequential_5/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/conv2d_transpose_13/stack/2?
(sequential_5/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_5/conv2d_transpose_13/stack/3?
&sequential_5/conv2d_transpose_13/stackPack7sequential_5/conv2d_transpose_13/strided_slice:output:01sequential_5/conv2d_transpose_13/stack/1:output:01sequential_5/conv2d_transpose_13/stack/2:output:01sequential_5/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_13/stack?
6sequential_5/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_5/conv2d_transpose_13/strided_slice_1/stack?
8sequential_5/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_13/strided_slice_1/stack_1?
8sequential_5/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_13/strided_slice_1/stack_2?
0sequential_5/conv2d_transpose_13/strided_slice_1StridedSlice/sequential_5/conv2d_transpose_13/stack:output:0?sequential_5/conv2d_transpose_13/strided_slice_1/stack:output:0Asequential_5/conv2d_transpose_13/strided_slice_1/stack_1:output:0Asequential_5/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_5/conv2d_transpose_13/strided_slice_1?
@sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_5_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02B
@sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
1sequential_5/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput/sequential_5/conv2d_transpose_13/stack:output:0Hsequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:03sequential_5/leaky_re_lu_28/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
23
1sequential_5/conv2d_transpose_13/conv2d_transpose?
2sequential_5/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_5/batch_normalization_14/ReadVariableOp?
4sequential_5/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype026
4sequential_5/batch_normalization_14/ReadVariableOp_1?
Csequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Csequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
Esequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02G
Esequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
4sequential_5/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3:sequential_5/conv2d_transpose_13/conv2d_transpose:output:0:sequential_5/batch_normalization_14/ReadVariableOp:value:0<sequential_5/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 26
4sequential_5/batch_normalization_14/FusedBatchNormV3?
%sequential_5/leaky_re_lu_29/LeakyRelu	LeakyRelu8sequential_5/batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2'
%sequential_5/leaky_re_lu_29/LeakyRelu?
&sequential_5/conv2d_transpose_14/ShapeShape3sequential_5/leaky_re_lu_29/LeakyRelu:activations:0*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_14/Shape?
4sequential_5/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_5/conv2d_transpose_14/strided_slice/stack?
6sequential_5/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_14/strided_slice/stack_1?
6sequential_5/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_14/strided_slice/stack_2?
.sequential_5/conv2d_transpose_14/strided_sliceStridedSlice/sequential_5/conv2d_transpose_14/Shape:output:0=sequential_5/conv2d_transpose_14/strided_slice/stack:output:0?sequential_5/conv2d_transpose_14/strided_slice/stack_1:output:0?sequential_5/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_5/conv2d_transpose_14/strided_slice?
(sequential_5/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_5/conv2d_transpose_14/stack/1?
(sequential_5/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_5/conv2d_transpose_14/stack/2?
(sequential_5/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_5/conv2d_transpose_14/stack/3?
&sequential_5/conv2d_transpose_14/stackPack7sequential_5/conv2d_transpose_14/strided_slice:output:01sequential_5/conv2d_transpose_14/stack/1:output:01sequential_5/conv2d_transpose_14/stack/2:output:01sequential_5/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_14/stack?
6sequential_5/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_5/conv2d_transpose_14/strided_slice_1/stack?
8sequential_5/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_14/strided_slice_1/stack_1?
8sequential_5/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_14/strided_slice_1/stack_2?
0sequential_5/conv2d_transpose_14/strided_slice_1StridedSlice/sequential_5/conv2d_transpose_14/stack:output:0?sequential_5/conv2d_transpose_14/strided_slice_1/stack:output:0Asequential_5/conv2d_transpose_14/strided_slice_1/stack_1:output:0Asequential_5/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_5/conv2d_transpose_14/strided_slice_1?
@sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_5_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02B
@sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
1sequential_5/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput/sequential_5/conv2d_transpose_14/stack:output:0Hsequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:03sequential_5/leaky_re_lu_29/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
23
1sequential_5/conv2d_transpose_14/conv2d_transpose?
2sequential_5/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_5/batch_normalization_15/ReadVariableOp?
4sequential_5/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_5/batch_normalization_15/ReadVariableOp_1?
Csequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
Esequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
4sequential_5/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3:sequential_5/conv2d_transpose_14/conv2d_transpose:output:0:sequential_5/batch_normalization_15/ReadVariableOp:value:0<sequential_5/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 26
4sequential_5/batch_normalization_15/FusedBatchNormV3?
%sequential_5/leaky_re_lu_30/LeakyRelu	LeakyRelu8sequential_5/batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2'
%sequential_5/leaky_re_lu_30/LeakyRelu?
&sequential_5/conv2d_transpose_15/ShapeShape3sequential_5/leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_15/Shape?
4sequential_5/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_5/conv2d_transpose_15/strided_slice/stack?
6sequential_5/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_15/strided_slice/stack_1?
6sequential_5/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_5/conv2d_transpose_15/strided_slice/stack_2?
.sequential_5/conv2d_transpose_15/strided_sliceStridedSlice/sequential_5/conv2d_transpose_15/Shape:output:0=sequential_5/conv2d_transpose_15/strided_slice/stack:output:0?sequential_5/conv2d_transpose_15/strided_slice/stack_1:output:0?sequential_5/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_5/conv2d_transpose_15/strided_slice?
(sequential_5/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_5/conv2d_transpose_15/stack/1?
(sequential_5/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(sequential_5/conv2d_transpose_15/stack/2?
(sequential_5/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_5/conv2d_transpose_15/stack/3?
&sequential_5/conv2d_transpose_15/stackPack7sequential_5/conv2d_transpose_15/strided_slice:output:01sequential_5/conv2d_transpose_15/stack/1:output:01sequential_5/conv2d_transpose_15/stack/2:output:01sequential_5/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/conv2d_transpose_15/stack?
6sequential_5/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_5/conv2d_transpose_15/strided_slice_1/stack?
8sequential_5/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_15/strided_slice_1/stack_1?
8sequential_5/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_5/conv2d_transpose_15/strided_slice_1/stack_2?
0sequential_5/conv2d_transpose_15/strided_slice_1StridedSlice/sequential_5/conv2d_transpose_15/stack:output:0?sequential_5/conv2d_transpose_15/strided_slice_1/stack:output:0Asequential_5/conv2d_transpose_15/strided_slice_1/stack_1:output:0Asequential_5/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_5/conv2d_transpose_15/strided_slice_1?
@sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpIsequential_5_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02B
@sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
1sequential_5/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput/sequential_5/conv2d_transpose_15/stack:output:0Hsequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:03sequential_5/leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
23
1sequential_5/conv2d_transpose_15/conv2d_transpose?
%sequential_5/conv2d_transpose_15/TanhTanh:sequential_5/conv2d_transpose_15/conv2d_transpose:output:0*
T0*/
_output_shapes
:?????????@@2'
%sequential_5/conv2d_transpose_15/Tanh?
IdentityIdentity)sequential_5/conv2d_transpose_15/Tanh:y:0=^sequential_5/batch_normalization_12/batchnorm/ReadVariableOp?^sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1?^sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2A^sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOpD^sequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_13/ReadVariableOp5^sequential_5/batch_normalization_13/ReadVariableOp_1D^sequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_14/ReadVariableOp5^sequential_5/batch_normalization_14/ReadVariableOp_1D^sequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_15/ReadVariableOp5^sequential_5/batch_normalization_15/ReadVariableOp_1A^sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOpA^sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOpA^sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOpA^sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^sequential_5/dense_7/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2|
<sequential_5/batch_normalization_12/batchnorm/ReadVariableOp<sequential_5/batch_normalization_12/batchnorm/ReadVariableOp2?
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_1>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_12?
>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_2>sequential_5/batch_normalization_12/batchnorm/ReadVariableOp_22?
@sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOp@sequential_5/batch_normalization_12/batchnorm/mul/ReadVariableOp2?
Csequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_13/ReadVariableOp2sequential_5/batch_normalization_13/ReadVariableOp2l
4sequential_5/batch_normalization_13/ReadVariableOp_14sequential_5/batch_normalization_13/ReadVariableOp_12?
Csequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_14/ReadVariableOp2sequential_5/batch_normalization_14/ReadVariableOp2l
4sequential_5/batch_normalization_14/ReadVariableOp_14sequential_5/batch_normalization_14/ReadVariableOp_12?
Csequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_15/ReadVariableOp2sequential_5/batch_normalization_15/ReadVariableOp2l
4sequential_5/batch_normalization_15/ReadVariableOp_14sequential_5/batch_normalization_15/ReadVariableOp_12?
@sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOp@sequential_5/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2?
@sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOp@sequential_5/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2?
@sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOp@sequential_5/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2?
@sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp@sequential_5/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*sequential_5/dense_7/MatMul/ReadVariableOp*sequential_5/dense_7/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
?*
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25688

inputs7
'assignmovingavg_readvariableop_resource:
??9
)assignmovingavg_1_readvariableop_resource:
??5
%batchnorm_mul_readvariableop_resource:
??1
!batchnorm_readvariableop_resource:
??
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2
moments/mean~
moments/StopGradientStopGradientmoments/mean:output:0*
T0* 
_output_shapes
:
??2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*)
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes

:??*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes

:??*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes

:??2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:??2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/mul_1}
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes

:??2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25761

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_25627

inputs2
matmul_readvariableop_resource:
d??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul~
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_13_layer_call_fn_25792

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_241862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25851

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?N
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25115
dense_7_input!
dense_7_25058:
d??,
batch_normalization_12_25061:
??,
batch_normalization_12_25063:
??,
batch_normalization_12_25065:
??,
batch_normalization_12_25067:
??5
conv2d_transpose_12_25072:??+
batch_normalization_13_25075:	?+
batch_normalization_13_25077:	?+
batch_normalization_13_25079:	?+
batch_normalization_13_25081:	?5
conv2d_transpose_13_25085:??+
batch_normalization_14_25088:	?+
batch_normalization_14_25090:	?+
batch_normalization_14_25092:	?+
batch_normalization_14_25094:	?4
conv2d_transpose_14_25098:@?*
batch_normalization_15_25101:@*
batch_normalization_15_25103:@*
batch_normalization_15_25105:@*
batch_normalization_15_25107:@3
conv2d_transpose_15_25111:@
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_25058*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_246742!
dense_7/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_12_25061batch_normalization_12_25063batch_normalization_12_25065batch_normalization_12_25067*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2398720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_246922 
leaky_re_lu_27/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall'leaky_re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_247082
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_25072*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_241562-
+conv2d_transpose_12/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0batch_normalization_13_25075batch_normalization_13_25077batch_normalization_13_25079batch_normalization_13_25081*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2418620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_247272 
leaky_re_lu_28/PartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_28/PartitionedCall:output:0conv2d_transpose_13_25085*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_243212-
+conv2d_transpose_13/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0batch_normalization_14_25088batch_normalization_14_25090batch_normalization_14_25092batch_normalization_14_25094*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2435120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_247462 
leaky_re_lu_29/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_transpose_14_25098*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_244862-
+conv2d_transpose_14/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0batch_normalization_15_25101batch_normalization_15_25103batch_normalization_15_25105batch_normalization_15_25107*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2451620
.batch_normalization_15/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_247652 
leaky_re_lu_30/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_transpose_15_25111*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_246522-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
?
?
6__inference_batch_normalization_15_layer_call_fn_25936

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_245162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_25055
dense_7_input
unknown:
d??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
	unknown_3:
??%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?%

unknown_14:@?

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*/
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_249632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
?
J
.__inference_leaky_re_lu_29_layer_call_fn_25887

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_247462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_24708

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25833

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_27_layer_call_fn_25724

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_246922
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_14_layer_call_fn_25877

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_243952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_25224
dense_7_input
unknown:
d??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
	unknown_3:
??%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?%

unknown_14:@?

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_239632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25654

inputs1
!batchnorm_readvariableop_resource:
??5
%batchnorm_mul_readvariableop_resource:
??3
#batchnorm_readvariableop_1_resource:
??3
#batchnorm_readvariableop_2_resource:
??
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes

:??2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:??2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes

:??2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_25573

inputs
unknown:
d??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
	unknown_3:
??%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?%

unknown_14:@?

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_247712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_24727

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_25719

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_24486

inputsC
(conv2d_transpose_readvariableop_resource:@?
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_14_layer_call_fn_24494

inputs"
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_244862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24395

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_24674

inputs2
matmul_readvariableop_resource:
d??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul~
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?!
?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_24652

inputsB
(conv2d_transpose_readvariableop_resource:@
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose{
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
E
)__inference_reshape_3_layer_call_fn_25743

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_247082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_25954

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_23987

inputs1
!batchnorm_readvariableop_resource:
??5
%batchnorm_mul_readvariableop_resource:
??3
#batchnorm_readvariableop_1_resource:
??3
#batchnorm_readvariableop_2_resource:
??
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes

:??2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:??2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes

:??2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_13_layer_call_fn_24329

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_243212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_24746

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_12_layer_call_fn_25714

inputs
unknown:
??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_240472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_12_layer_call_fn_25701

inputs
unknown:
??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_239872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_28_layer_call_fn_25815

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_247272
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?*
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_24047

inputs7
'assignmovingavg_readvariableop_resource:
??9
)assignmovingavg_1_readvariableop_resource:
??5
%batchnorm_mul_readvariableop_resource:
??1
!batchnorm_readvariableop_resource:
??
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2
moments/mean~
moments/StopGradientStopGradientmoments/mean:output:0*
T0* 
_output_shapes
:
??2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*)
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes

:??*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes

:??*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:??2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes

:??2
batchnorm/adde
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes

:??2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2
batchnorm/mulx
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/mul_1}
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes

:??2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24560

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_30_layer_call_fn_25959

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_247652
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25923

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_24516

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?M
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_24963

inputs!
dense_7_24906:
d??,
batch_normalization_12_24909:
??,
batch_normalization_12_24911:
??,
batch_normalization_12_24913:
??,
batch_normalization_12_24915:
??5
conv2d_transpose_12_24920:??+
batch_normalization_13_24923:	?+
batch_normalization_13_24925:	?+
batch_normalization_13_24927:	?+
batch_normalization_13_24929:	?5
conv2d_transpose_13_24933:??+
batch_normalization_14_24936:	?+
batch_normalization_14_24938:	?+
batch_normalization_14_24940:	?+
batch_normalization_14_24942:	?4
conv2d_transpose_14_24946:@?*
batch_normalization_15_24949:@*
batch_normalization_15_24951:@*
batch_normalization_15_24953:@*
batch_normalization_15_24955:@3
conv2d_transpose_15_24959:@
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_24906*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_246742!
dense_7/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_12_24909batch_normalization_12_24911batch_normalization_12_24913batch_normalization_12_24915*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2404720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_246922 
leaky_re_lu_27/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall'leaky_re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_247082
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_24920*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_241562-
+conv2d_transpose_12/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0batch_normalization_13_24923batch_normalization_13_24925batch_normalization_13_24927batch_normalization_13_24929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2423020
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_247272 
leaky_re_lu_28/PartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_28/PartitionedCall:output:0conv2d_transpose_13_24933*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_243212-
+conv2d_transpose_13/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0batch_normalization_14_24936batch_normalization_14_24938batch_normalization_14_24940batch_normalization_14_24942*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2439520
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_247462 
leaky_re_lu_29/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_transpose_14_24946*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_244862-
+conv2d_transpose_14/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0batch_normalization_15_24949batch_normalization_15_24951batch_normalization_15_24953batch_normalization_15_24955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2456020
.batch_normalization_15/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_247652 
leaky_re_lu_30/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_transpose_15_24959*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_246522-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24186

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25526

inputs:
&dense_7_matmul_readvariableop_resource:
d??N
>batch_normalization_12_assignmovingavg_readvariableop_resource:
??P
@batch_normalization_12_assignmovingavg_1_readvariableop_resource:
??L
<batch_normalization_12_batchnorm_mul_readvariableop_resource:
??H
8batch_normalization_12_batchnorm_readvariableop_resource:
??X
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:??=
.batch_normalization_13_readvariableop_resource:	??
0batch_normalization_13_readvariableop_1_resource:	?N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	?X
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource:??=
.batch_normalization_14_readvariableop_resource:	??
0batch_normalization_14_readvariableop_1_resource:	?N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource:@?<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:@
identity??&batch_normalization_12/AssignMovingAvg?5batch_normalization_12/AssignMovingAvg/ReadVariableOp?(batch_normalization_12/AssignMovingAvg_1?7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_12/batchnorm/ReadVariableOp?3batch_normalization_12/batchnorm/mul/ReadVariableOp?%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?%batch_normalization_14/AssignNewValue?'batch_normalization_14/AssignNewValue_1?6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_14/ReadVariableOp?'batch_normalization_14/ReadVariableOp_1?%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_7/MatMul?
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_12/moments/mean/reduction_indices?
#batch_normalization_12/moments/meanMeandense_7/MatMul:product:0>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2%
#batch_normalization_12/moments/mean?
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0* 
_output_shapes
:
??2-
+batch_normalization_12/moments/StopGradient?
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencedense_7/MatMul:product:04batch_normalization_12/moments/StopGradient:output:0*
T0*)
_output_shapes
:???????????22
0batch_normalization_12/moments/SquaredDifference?
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_12/moments/variance/reduction_indices?
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0* 
_output_shapes
:
??*
	keep_dims(2)
'batch_normalization_12/moments/variance?
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2(
&batch_normalization_12/moments/Squeeze?
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes

:??*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1?
,batch_normalization_12/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_12/AssignMovingAvg/decay?
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource*
_output_shapes

:??*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOp?
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*
_output_shapes

:??2,
*batch_normalization_12/AssignMovingAvg/sub?
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:05batch_normalization_12/AssignMovingAvg/decay:output:0*
T0*
_output_shapes

:??2,
*batch_normalization_12/AssignMovingAvg/mul?
&batch_normalization_12/AssignMovingAvgAssignSubVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_12/AssignMovingAvg?
.batch_normalization_12/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_12/AssignMovingAvg_1/decay?
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource*
_output_shapes

:??*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*
_output_shapes

:??2.
,batch_normalization_12/AssignMovingAvg_1/sub?
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:07batch_normalization_12/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes

:??2.
,batch_normalization_12/AssignMovingAvg_1/mul?
(batch_normalization_12/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_12/AssignMovingAvg_1?
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_12/batchnorm/add/y?
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/add?
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes

:??2(
&batch_normalization_12/batchnorm/Rsqrt?
3batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes

:??*
dtype025
3batch_normalization_12/batchnorm/mul/ReadVariableOp?
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:0;batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/mul?
&batch_normalization_12/batchnorm/mul_1Muldense_7/MatMul:product:0(batch_normalization_12/batchnorm/mul:z:0*
T0*)
_output_shapes
:???????????2(
&batch_normalization_12/batchnorm/mul_1?
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes

:??2(
&batch_normalization_12/batchnorm/mul_2?
/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes

:??*
dtype021
/batch_normalization_12/batchnorm/ReadVariableOp?
$batch_normalization_12/batchnorm/subSub7batch_normalization_12/batchnorm/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes

:??2&
$batch_normalization_12/batchnorm/sub?
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*)
_output_shapes
:???????????2(
&batch_normalization_12/batchnorm/add_1?
leaky_re_lu_27/LeakyRelu	LeakyRelu*batch_normalization_12/batchnorm/add_1:z:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_27/LeakyRelux
reshape_3/ShapeShape&leaky_re_lu_27/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2y
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_3/Reshape/shape/3?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshape&leaky_re_lu_27/LeakyRelu:activations:0 reshape_3/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_3/Reshape?
conv2d_transpose_12/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2}
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_13/ReadVariableOp?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_13/ReadVariableOp_1?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_12/conv2d_transpose:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_13/FusedBatchNormV3?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1?
leaky_re_lu_28/LeakyRelu	LeakyRelu+batch_normalization_13/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_28/LeakyRelu?
conv2d_transpose_13/ShapeShape&leaky_re_lu_28/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2}
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_28/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_14/ReadVariableOp?
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_14/ReadVariableOp_1?
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_13/conv2d_transpose:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_14/FusedBatchNormV3?
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValue?
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1?
leaky_re_lu_29/LeakyRelu	LeakyRelu+batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>2
leaky_re_lu_29/LeakyRelu?
conv2d_transpose_14/ShapeShape&leaky_re_lu_29/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_29/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_14/conv2d_transpose:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
leaky_re_lu_30/LeakyRelu	LeakyRelu+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????  @*
alpha%???>2
leaky_re_lu_30/LeakyRelu?
conv2d_transpose_15/ShapeShape&leaky_re_lu_30/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_30/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
conv2d_transpose_15/TanhTanh-conv2d_transpose_15/conv2d_transpose:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_15/Tanh?
IdentityIdentityconv2d_transpose_15/Tanh:y:0'^batch_normalization_12/AssignMovingAvg6^batch_normalization_12/AssignMovingAvg/ReadVariableOp)^batch_normalization_12/AssignMovingAvg_18^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_12/batchnorm/ReadVariableOp4^batch_normalization_12/batchnorm/mul/ReadVariableOp&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_14^conv2d_transpose_12/conv2d_transpose/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_12/AssignMovingAvg&batch_normalization_12/AssignMovingAvg2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_12/AssignMovingAvg_1(batch_normalization_12/AssignMovingAvg_12r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_12/batchnorm/ReadVariableOp/batch_normalization_12/batchnorm/ReadVariableOp2j
3batch_normalization_12/batchnorm/mul/ReadVariableOp3batch_normalization_12/batchnorm/mul/ReadVariableOp2N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_24156

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_24321

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_15_layer_call_fn_24660

inputs!
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_246522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_12_layer_call_fn_24164

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_241562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25779

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?M
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_24771

inputs!
dense_7_24675:
d??,
batch_normalization_12_24678:
??,
batch_normalization_12_24680:
??,
batch_normalization_12_24682:
??,
batch_normalization_12_24684:
??5
conv2d_transpose_12_24710:??+
batch_normalization_13_24713:	?+
batch_normalization_13_24715:	?+
batch_normalization_13_24717:	?+
batch_normalization_13_24719:	?5
conv2d_transpose_13_24729:??+
batch_normalization_14_24732:	?+
batch_normalization_14_24734:	?+
batch_normalization_14_24736:	?+
batch_normalization_14_24738:	?4
conv2d_transpose_14_24748:@?*
batch_normalization_15_24751:@*
batch_normalization_15_24753:@*
batch_normalization_15_24755:@*
batch_normalization_15_24757:@3
conv2d_transpose_15_24767:@
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_24675*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_246742!
dense_7/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_12_24678batch_normalization_12_24680batch_normalization_12_24682batch_normalization_12_24684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2398720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_246922 
leaky_re_lu_27/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall'leaky_re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_247082
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_24710*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_241562-
+conv2d_transpose_12/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0batch_normalization_13_24713batch_normalization_13_24715batch_normalization_13_24717batch_normalization_13_24719*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2418620
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_247272 
leaky_re_lu_28/PartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_28/PartitionedCall:output:0conv2d_transpose_13_24729*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_243212-
+conv2d_transpose_13/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0batch_normalization_14_24732batch_normalization_14_24734batch_normalization_14_24736batch_normalization_14_24738*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2435120
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_247462 
leaky_re_lu_29/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_transpose_14_24748*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_244862-
+conv2d_transpose_14/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0batch_normalization_15_24751batch_normalization_15_24753batch_normalization_15_24755batch_normalization_15_24757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2451620
.batch_normalization_15/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_247652 
leaky_re_lu_30/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_transpose_15_24767*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_246522-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_24692

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?N
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25175
dense_7_input!
dense_7_25118:
d??,
batch_normalization_12_25121:
??,
batch_normalization_12_25123:
??,
batch_normalization_12_25125:
??,
batch_normalization_12_25127:
??5
conv2d_transpose_12_25132:??+
batch_normalization_13_25135:	?+
batch_normalization_13_25137:	?+
batch_normalization_13_25139:	?+
batch_normalization_13_25141:	?5
conv2d_transpose_13_25145:??+
batch_normalization_14_25148:	?+
batch_normalization_14_25150:	?+
batch_normalization_14_25152:	?+
batch_normalization_14_25154:	?4
conv2d_transpose_14_25158:@?*
batch_normalization_15_25161:@*
batch_normalization_15_25163:@*
batch_normalization_15_25165:@*
batch_normalization_15_25167:@3
conv2d_transpose_15_25171:@
identity??.batch_normalization_12/StatefulPartitionedCall?.batch_normalization_13/StatefulPartitionedCall?.batch_normalization_14/StatefulPartitionedCall?.batch_normalization_15/StatefulPartitionedCall?+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_25118*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_246742!
dense_7/StatefulPartitionedCall?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_12_25121batch_normalization_12_25123batch_normalization_12_25125batch_normalization_12_25127*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_2404720
.batch_normalization_12/StatefulPartitionedCall?
leaky_re_lu_27/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_246922 
leaky_re_lu_27/PartitionedCall?
reshape_3/PartitionedCallPartitionedCall'leaky_re_lu_27/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_247082
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_25132*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_241562-
+conv2d_transpose_12/StatefulPartitionedCall?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0batch_normalization_13_25135batch_normalization_13_25137batch_normalization_13_25139batch_normalization_13_25141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_2423020
.batch_normalization_13/StatefulPartitionedCall?
leaky_re_lu_28/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_247272 
leaky_re_lu_28/PartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_28/PartitionedCall:output:0conv2d_transpose_13_25145*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_243212-
+conv2d_transpose_13/StatefulPartitionedCall?
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0batch_normalization_14_25148batch_normalization_14_25150batch_normalization_14_25152batch_normalization_14_25154*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_2439520
.batch_normalization_14/StatefulPartitionedCall?
leaky_re_lu_29/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_247462 
leaky_re_lu_29/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_29/PartitionedCall:output:0conv2d_transpose_14_25158*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_244862-
+conv2d_transpose_14/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0batch_normalization_15_25161batch_normalization_15_25163batch_normalization_15_25165batch_normalization_15_25167*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2456020
.batch_normalization_15/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_247652 
leaky_re_lu_30/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_transpose_15_25171*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_246522-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:V R
'
_output_shapes
:?????????d
'
_user_specified_namedense_7_input
?
?
,__inference_sequential_5_layer_call_fn_25620

inputs
unknown:
d??
	unknown_0:
??
	unknown_1:
??
	unknown_2:
??
	unknown_3:
??%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?%

unknown_14:@?

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*/
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_249632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????d: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_24765

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+???????????????????????????@*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?`
?
!__inference__traced_restore_26118
file_prefix3
assignvariableop_dense_7_kernel:
d???
/assignvariableop_1_batch_normalization_12_gamma:
??>
.assignvariableop_2_batch_normalization_12_beta:
??E
5assignvariableop_3_batch_normalization_12_moving_mean:
??I
9assignvariableop_4_batch_normalization_12_moving_variance:
??I
-assignvariableop_5_conv2d_transpose_12_kernel:??>
/assignvariableop_6_batch_normalization_13_gamma:	?=
.assignvariableop_7_batch_normalization_13_beta:	?D
5assignvariableop_8_batch_normalization_13_moving_mean:	?H
9assignvariableop_9_batch_normalization_13_moving_variance:	?J
.assignvariableop_10_conv2d_transpose_13_kernel:???
0assignvariableop_11_batch_normalization_14_gamma:	?>
/assignvariableop_12_batch_normalization_14_beta:	?E
6assignvariableop_13_batch_normalization_14_moving_mean:	?I
:assignvariableop_14_batch_normalization_14_moving_variance:	?I
.assignvariableop_15_conv2d_transpose_14_kernel:@?>
0assignvariableop_16_batch_normalization_15_gamma:@=
/assignvariableop_17_batch_normalization_15_beta:@D
6assignvariableop_18_batch_normalization_15_moving_mean:@H
:assignvariableop_19_batch_normalization_15_moving_variance:@H
.assignvariableop_20_conv2d_transpose_15_kernel:@
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_12_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_12_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_12_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_12_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_conv2d_transpose_12_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_13_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_13_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_13_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_13_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv2d_transpose_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_14_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_14_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_14_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_14_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_conv2d_transpose_14_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_15_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_15_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_15_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_15_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_conv2d_transpose_15_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21?
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_24351

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
}
'__inference_dense_7_layer_call_fn_25634

inputs
unknown:
d??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_246742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25905

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_24230

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_14_layer_call_fn_25864

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_243512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_25949

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_245602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_25810

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?8
?
__inference__traced_save_26045
file_prefix-
)savev2_dense_7_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_12_kernel_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_13_kernel_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_14_kernel_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_15_kernel_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop5savev2_conv2d_transpose_12_kernel_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop5savev2_conv2d_transpose_13_kernel_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop5savev2_conv2d_transpose_14_kernel_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop5savev2_conv2d_transpose_15_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
d??:??:??:??:??:??:?:?:?:?:??:?:?:?:?:@?:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
d??:"

_output_shapes

:??:"

_output_shapes

:??:"

_output_shapes

:??:"

_output_shapes

:??:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:

_output_shapes
: 
?
e
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_25882

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
dense_7_input6
serving_default_dense_7_input:0?????????dO
conv2d_transpose_158
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?y
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?t
_tf_keras_sequential?t{"name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_27", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 512]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_28", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_29", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100]}, "float32", "dense_7_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_27", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 9}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 512]}}, "shared_object_id": 10}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_28", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 19}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 22}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 27}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_29", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 28}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 31}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 36}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 37}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 40}]}}}
?	

kernel
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 32768}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32768]}}
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "leaky_re_lu_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_27", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 9}
?
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [8, 8, 512]}}, "shared_object_id": 10}
?

*kernel
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 512]}}
?

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "leaky_re_lu_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_28", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 19}
?

<kernel
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}
?

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "leaky_re_lu_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_29", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 28}
?

Nkernel
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?

Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 35}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "leaky_re_lu_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 37}
?

`kernel
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_transpose_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
 "
trackable_list_wrapper
~
0
1
2
*3
04
15
<6
B7
C8
N9
T10
U11
`12"
trackable_list_wrapper
?
0
1
2
3
4
*5
06
17
28
39
<10
B11
C12
D13
E14
N15
T16
U17
V18
W19
`20"
trackable_list_wrapper
?
enon_trainable_variables
regularization_losses

flayers
glayer_regularization_losses
hlayer_metrics
imetrics
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
": 
d??2dense_7/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
jnon_trainable_variables
regularization_losses

klayers
llayer_regularization_losses
mlayer_metrics
nmetrics
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*??2batch_normalization_12/gamma
+:)??2batch_normalization_12/beta
4:2?? (2"batch_normalization_12/moving_mean
8:6?? (2&batch_normalization_12/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
onon_trainable_variables
regularization_losses

players
qlayer_regularization_losses
rlayer_metrics
smetrics
trainable_variables
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables
"regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
xmetrics
#trainable_variables
$	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables
&regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
}metrics
'trainable_variables
(	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4??2conv2d_transpose_12/kernel
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
?
~non_trainable_variables
+regularization_losses

layers
 ?layer_regularization_losses
?layer_metrics
?metrics
,trainable_variables
-	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_13/gamma
*:(?2batch_normalization_13/beta
3:1? (2"batch_normalization_13/moving_mean
7:5? (2&batch_normalization_13/moving_variance
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
?
?non_trainable_variables
4regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
5trainable_variables
6	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
8regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
9trainable_variables
:	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
6:4??2conv2d_transpose_13/kernel
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
?
?non_trainable_variables
=regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
>trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_14/gamma
*:(?2batch_normalization_14/beta
3:1? (2"batch_normalization_14/moving_mean
7:5? (2&batch_normalization_14/moving_variance
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
?
?non_trainable_variables
Fregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Gtrainable_variables
H	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Jregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ktrainable_variables
L	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3@?2conv2d_transpose_14/kernel
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
?
?non_trainable_variables
Oregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ptrainable_variables
Q	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_15/gamma
):'@2batch_normalization_15/beta
2:0@ (2"batch_normalization_15/moving_mean
6:4@ (2&batch_normalization_15/moving_variance
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
?
?non_trainable_variables
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
Ytrainable_variables
Z	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
\regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
]trainable_variables
^	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2@2conv2d_transpose_15/kernel
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
?
?non_trainable_variables
aregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
btrainable_variables
c	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
1
22
33
D4
E5
V6
W7"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25368
G__inference_sequential_5_layer_call_and_return_conditional_losses_25526
G__inference_sequential_5_layer_call_and_return_conditional_losses_25115
G__inference_sequential_5_layer_call_and_return_conditional_losses_25175?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_5_layer_call_fn_24816
,__inference_sequential_5_layer_call_fn_25573
,__inference_sequential_5_layer_call_fn_25620
,__inference_sequential_5_layer_call_fn_25055?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_23963?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
dense_7_input?????????d
?2?
B__inference_dense_7_layer_call_and_return_conditional_losses_25627?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_7_layer_call_fn_25634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25654
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25688?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_12_layer_call_fn_25701
6__inference_batch_normalization_12_layer_call_fn_25714?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_25719?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_27_layer_call_fn_25724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_3_layer_call_and_return_conditional_losses_25738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_3_layer_call_fn_25743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_24156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
3__inference_conv2d_transpose_12_layer_call_fn_24164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25761
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25779?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_13_layer_call_fn_25792
6__inference_batch_normalization_13_layer_call_fn_25805?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_25810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_28_layer_call_fn_25815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_24321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
3__inference_conv2d_transpose_13_layer_call_fn_24329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25833
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25851?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_14_layer_call_fn_25864
6__inference_batch_normalization_14_layer_call_fn_25877?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_25882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_29_layer_call_fn_25887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_24486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
3__inference_conv2d_transpose_14_layer_call_fn_24494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25905
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25923?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_15_layer_call_fn_25936
6__inference_batch_normalization_15_layer_call_fn_25949?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_25954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_30_layer_call_fn_25959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_24652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
3__inference_conv2d_transpose_15_layer_call_fn_24660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?B?
#__inference_signature_wrapper_25224dense_7_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_23963?*0123<BCDENTUVW`6?3
,?)
'?$
dense_7_input?????????d
? "Q?N
L
conv2d_transpose_155?2
conv2d_transpose_15?????????@@?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25654f5?2
+?(
"?
inputs???????????
p 
? "'?$
?
0???????????
? ?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_25688f5?2
+?(
"?
inputs???????????
p
? "'?$
?
0???????????
? ?
6__inference_batch_normalization_12_layer_call_fn_25701Y5?2
+?(
"?
inputs???????????
p 
? "?????????????
6__inference_batch_normalization_12_layer_call_fn_25714Y5?2
+?(
"?
inputs???????????
p
? "?????????????
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25761?0123N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_25779?0123N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_13_layer_call_fn_25792?0123N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_13_layer_call_fn_25805?0123N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25833?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_25851?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_14_layer_call_fn_25864?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_14_layer_call_fn_25877?BCDEN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25905?TUVWM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_25923?TUVWM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_batch_normalization_15_layer_call_fn_25936?TUVWM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_15_layer_call_fn_25949?TUVWM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_24156?*J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
3__inference_conv2d_transpose_12_layer_call_fn_24164?*J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_24321?<J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
3__inference_conv2d_transpose_13_layer_call_fn_24329?<J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_24486?NJ?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_conv2d_transpose_14_layer_call_fn_24494?NJ?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_24652?`I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_15_layer_call_fn_24660?`I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
B__inference_dense_7_layer_call_and_return_conditional_losses_25627]/?,
%?"
 ?
inputs?????????d
? "'?$
?
0???????????
? {
'__inference_dense_7_layer_call_fn_25634P/?,
%?"
 ?
inputs?????????d
? "?????????????
I__inference_leaky_re_lu_27_layer_call_and_return_conditional_losses_25719\1?.
'?$
"?
inputs???????????
? "'?$
?
0???????????
? ?
.__inference_leaky_re_lu_27_layer_call_fn_25724O1?.
'?$
"?
inputs???????????
? "?????????????
I__inference_leaky_re_lu_28_layer_call_and_return_conditional_losses_25810?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_leaky_re_lu_28_layer_call_fn_25815?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_leaky_re_lu_29_layer_call_and_return_conditional_losses_25882?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_leaky_re_lu_29_layer_call_fn_25887?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_25954?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
.__inference_leaky_re_lu_30_layer_call_fn_25959I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
D__inference_reshape_3_layer_call_and_return_conditional_losses_25738c1?.
'?$
"?
inputs???????????
? ".?+
$?!
0??????????
? ?
)__inference_reshape_3_layer_call_fn_25743V1?.
'?$
"?
inputs???????????
? "!????????????
G__inference_sequential_5_layer_call_and_return_conditional_losses_25115?*0123<BCDENTUVW`>?;
4?1
'?$
dense_7_input?????????d
p 

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25175?*0123<BCDENTUVW`>?;
4?1
'?$
dense_7_input?????????d
p

 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25368*0123<BCDENTUVW`7?4
-?*
 ?
inputs?????????d
p 

 
? "-?*
#? 
0?????????@@
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_25526*0123<BCDENTUVW`7?4
-?*
 ?
inputs?????????d
p

 
? "-?*
#? 
0?????????@@
? ?
,__inference_sequential_5_layer_call_fn_24816?*0123<BCDENTUVW`>?;
4?1
'?$
dense_7_input?????????d
p 

 
? "2?/+????????????????????????????
,__inference_sequential_5_layer_call_fn_25055?*0123<BCDENTUVW`>?;
4?1
'?$
dense_7_input?????????d
p

 
? "2?/+????????????????????????????
,__inference_sequential_5_layer_call_fn_25573?*0123<BCDENTUVW`7?4
-?*
 ?
inputs?????????d
p 

 
? "2?/+????????????????????????????
,__inference_sequential_5_layer_call_fn_25620?*0123<BCDENTUVW`7?4
-?*
 ?
inputs?????????d
p

 
? "2?/+????????????????????????????
#__inference_signature_wrapper_25224?*0123<BCDENTUVW`G?D
? 
=?:
8
dense_7_input'?$
dense_7_input?????????d"Q?N
L
conv2d_transpose_155?2
conv2d_transpose_15?????????@@
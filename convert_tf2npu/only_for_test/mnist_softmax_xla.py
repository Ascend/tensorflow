Module(
|   body=[
|   |   Expr(
|   |   |   value=Str(
|   |   |   |   s='SimpleMNISTclassifierexamplewithJITXLAandtimelines.\n\nNote:PleaseseefurthercommentsintheBUILDfiletoinvokeXLA.\n'
|   |   |   )
|   |   ),
|   |   ImportFrom(
|   |   |   module='__future__',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='absolute_import',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   ImportFrom(
|   |   |   module='__future__',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='division',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   ImportFrom(
|   |   |   module='__future__',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='print_function',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   ImportFrom(
|   |   |   module='npu_bridge.npu_init',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='*',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   Import(
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='argparse',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ]
|   |   ),
|   |   Import(
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='sys',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ]
|   |   ),
|   |   Import(
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='tensorflow',
|   |   |   |   |   asname='tf'
|   |   |   |   )
|   |   |   ]
|   |   ),
|   |   ImportFrom(
|   |   |   module='tensorflow.examples.tutorials.mnist',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='input_data',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   ImportFrom(
|   |   |   module='tensorflow.python.client',
|   |   |   names=[
|   |   |   |   alias(
|   |   |   |   |   name='timeline',
|   |   |   |   |   asname=None
|   |   |   |   )
|   |   |   ],
|   |   |   level=0
|   |   ),
|   |   Assign(
|   |   |   targets=[
|   |   |   |   Name(
|   |   |   |   |   id='FLAGS',
|   |   |   |   |   ctx=Store(
|   |   |   |   |   |   
|   |   |   |   |   )
|   |   |   |   )
|   |   |   ],
|   |   |   value=NameConstant(
|   |   |   |   value=None
|   |   |   )
|   |   ),
|   |   FunctionDef(
|   |   |   name='main',
|   |   |   args=arguments(
|   |   |   |   args=[
|   |   |   |   |   arg(
|   |   |   |   |   |   arg='_',
|   |   |   |   |   |   annotation=None
|   |   |   |   |   )
|   |   |   |   ],
|   |   |   |   vararg=None,
|   |   |   |   kwonlyargs=[
|   |   |   |   |   
|   |   |   |   ],
|   |   |   |   kw_defaults=[
|   |   |   |   |   
|   |   |   |   ],
|   |   |   |   kwarg=None,
|   |   |   |   defaults=[
|   |   |   |   |   
|   |   |   |   ]
|   |   |   ),
|   |   |   body=[
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='mnist',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='input_data',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='read_data_sets',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='FLAGS',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='data_dir',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='x',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='placeholder',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='float32',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   List(
|   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   NameConstant(
|   |   |   |   |   |   |   |   |   |   value=None
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   n=784
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='w',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='Variable',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='zeros',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   List(
|   |   |   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   |   |   n=784
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   |   |   n=10
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='b',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='Variable',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='zeros',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   List(
|   |   |   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   |   |   n=10
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='y',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=BinOp(
|   |   |   |   |   |   left=Call(
|   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='matmul',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   id='x',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   id='w',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   ]
|   |   |   |   |   |   ),
|   |   |   |   |   |   op=Add(
|   |   |   |   |   |   |   
|   |   |   |   |   |   ),
|   |   |   |   |   |   right=Name(
|   |   |   |   |   |   |   id='b',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='placeholder',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='int64',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   List(
|   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   NameConstant(
|   |   |   |   |   |   |   |   |   |   value=None
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='cross_entropy',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='losses',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='sparse_softmax_cross_entropy',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='labels',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='logits',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='y',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='train_step',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   func=Name(
|   |   |   |   |   |   |   |   |   id='npu_distributed_optimizer_wrapper',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   attr='train',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   attr='GradientDescentOptimizer',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   |   |   n=0.5
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='minimize',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   id='cross_entropy',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='config',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='ConfigProto',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='jit_level',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Num(
|   |   |   |   |   |   n=0
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   If(
|   |   |   |   |   test=Attribute(
|   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   id='FLAGS',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   attr='xla',
|   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   
|   |   |   |   |   |   )
|   |   |   |   |   ),
|   |   |   |   |   body=[
|   |   |   |   |   |   Assign(
|   |   |   |   |   |   |   targets=[
|   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   id='jit_level',
|   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='OptimizerOptions',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='ON_1',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   orelse=[
|   |   |   |   |   |   
|   |   |   |   |   ]
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='config',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='graph_options',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='optimizer_options',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='global_jit_level',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Name(
|   |   |   |   |   |   id='jit_level',
|   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   
|   |   |   |   |   |   )
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='run_metadata',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='RunMetadata',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='compat',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='v1',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='Session',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='config',
|   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   func=Name(
|   |   |   |   |   |   |   |   |   |   id='npu_config_proto',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   arg='config_proto',
|   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='config',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='global_variables_initializer',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='run',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='session',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='train_loops',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Num(
|   |   |   |   |   |   n=1000
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   For(
|   |   |   |   |   target=Name(
|   |   |   |   |   |   id='i',
|   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   
|   |   |   |   |   |   )
|   |   |   |   |   ),
|   |   |   |   |   iter=Call(
|   |   |   |   |   |   func=Name(
|   |   |   |   |   |   |   id='range',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   id='train_loops',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   ),
|   |   |   |   |   body=[
|   |   |   |   |   |   Assign(
|   |   |   |   |   |   |   targets=[
|   |   |   |   |   |   |   |   Tuple(
|   |   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   id='batch_xs',
|   |   |   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   id='batch_ys',
|   |   |   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   id='mnist',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   attr='train',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='next_batch',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   n=100
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   If(
|   |   |   |   |   |   |   test=Compare(
|   |   |   |   |   |   |   |   left=Name(
|   |   |   |   |   |   |   |   |   id='i',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   ops=[
|   |   |   |   |   |   |   |   |   Eq(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   comparators=[
|   |   |   |   |   |   |   |   |   BinOp(
|   |   |   |   |   |   |   |   |   |   left=Name(
|   |   |   |   |   |   |   |   |   |   |   id='train_loops',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   op=Sub(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   right=Num(
|   |   |   |   |   |   |   |   |   |   |   n=1
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   body=[
|   |   |   |   |   |   |   |   Expr(
|   |   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   attr='run',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='train_step',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   arg='feed_dict',
|   |   |   |   |   |   |   |   |   |   |   |   value=Dict(
|   |   |   |   |   |   |   |   |   |   |   |   |   keys=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='x',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   |   values=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='batch_xs',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='batch_ys',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   arg='options',
|   |   |   |   |   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='RunOptions',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   arg='trace_level',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='RunOptions',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='FULL_TRACE',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   arg='run_metadata',
|   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   id='run_metadata',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   Assign(
|   |   |   |   |   |   |   |   |   targets=[
|   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   id='trace',
|   |   |   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='timeline',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   attr='Timeline',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   arg='step_stats',
|   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   id='run_metadata',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   attr='step_stats',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   With(
|   |   |   |   |   |   |   |   |   items=[
|   |   |   |   |   |   |   |   |   |   withitem(
|   |   |   |   |   |   |   |   |   |   |   context_expr=Call(
|   |   |   |   |   |   |   |   |   |   |   |   func=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   id='open',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   |   |   Str(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   s='/tmp/timeline.ctf.json'
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   Str(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   s='w'
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   optional_vars=Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='trace_file',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   body=[
|   |   |   |   |   |   |   |   |   |   Expr(
|   |   |   |   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   id='trace_file',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   attr='write',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='trace',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='generate_chrome_trace_format',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   orelse=[
|   |   |   |   |   |   |   |   Expr(
|   |   |   |   |   |   |   |   |   value=Call(
|   |   |   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   attr='run',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   id='train_step',
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   |   |   arg='feed_dict',
|   |   |   |   |   |   |   |   |   |   |   |   value=Dict(
|   |   |   |   |   |   |   |   |   |   |   |   |   keys=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='x',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   |   |   values=[
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='batch_xs',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='batch_ys',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ]
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   orelse=[
|   |   |   |   |   |   
|   |   |   |   |   ]
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='correct_prediction',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='equal',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='argmax',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   id='y',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   Num(
|   |   |   |   |   |   |   |   |   |   n=1
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='accuracy',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='reduce_mean',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='cast',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   id='correct_prediction',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   attr='float32',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Name(
|   |   |   |   |   |   |   id='print',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Call(
|   |   |   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   attr='run',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   args=[
|   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   id='accuracy',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   |   |   arg='feed_dict',
|   |   |   |   |   |   |   |   |   |   value=Dict(
|   |   |   |   |   |   |   |   |   |   |   keys=[
|   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   id='x',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   id='y_',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   |   values=[
|   |   |   |   |   |   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='mnist',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='test',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   attr='images',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   id='mnist',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   |   attr='test',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   attr='labels',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ]
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='sess',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='close',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   )
|   |   |   ],
|   |   |   decorator_list=[
|   |   |   |   
|   |   |   ],
|   |   |   returns=None
|   |   ),
|   |   If(
|   |   |   test=Compare(
|   |   |   |   left=Name(
|   |   |   |   |   id='__name__',
|   |   |   |   |   ctx=Load(
|   |   |   |   |   |   
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   ops=[
|   |   |   |   |   Eq(
|   |   |   |   |   |   
|   |   |   |   |   )
|   |   |   |   ],
|   |   |   |   comparators=[
|   |   |   |   |   Str(
|   |   |   |   |   |   s='__main__'
|   |   |   |   |   )
|   |   |   |   ]
|   |   |   ),
|   |   |   body=[
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Name(
|   |   |   |   |   |   |   id='parser',
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='argparse',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='ArgumentParser',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='parser',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='add_argument',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Str(
|   |   |   |   |   |   |   |   s='--data_dir'
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='type',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='str',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='default',
|   |   |   |   |   |   |   |   value=Str(
|   |   |   |   |   |   |   |   |   s='/tmp/tensorflow/mnist/input_data'
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='help',
|   |   |   |   |   |   |   |   value=Str(
|   |   |   |   |   |   |   |   |   s='Directoryforstoringinputdata'
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='parser',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='add_argument',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   Str(
|   |   |   |   |   |   |   |   s='--xla'
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='type',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='bool',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='default',
|   |   |   |   |   |   |   |   value=NameConstant(
|   |   |   |   |   |   |   |   |   value=True
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='help',
|   |   |   |   |   |   |   |   value=Str(
|   |   |   |   |   |   |   |   |   s='TurnxlaviaJITon'
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Assign(
|   |   |   |   |   targets=[
|   |   |   |   |   |   Tuple(
|   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   id='FLAGS',
|   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   Name(
|   |   |   |   |   |   |   |   |   id='unparsed',
|   |   |   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   ctx=Store(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   )
|   |   |   |   |   ],
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   id='parser',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='parse_known_args',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   ),
|   |   |   |   Expr(
|   |   |   |   |   value=Call(
|   |   |   |   |   |   func=Attribute(
|   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='tf',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   attr='app',
|   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   attr='run',
|   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ),
|   |   |   |   |   |   args=[
|   |   |   |   |   |   |   
|   |   |   |   |   |   ],
|   |   |   |   |   |   keywords=[
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='main',
|   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   id='main',
|   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   keyword(
|   |   |   |   |   |   |   |   arg='argv',
|   |   |   |   |   |   |   |   value=BinOp(
|   |   |   |   |   |   |   |   |   left=List(
|   |   |   |   |   |   |   |   |   |   elts=[
|   |   |   |   |   |   |   |   |   |   |   Subscript(
|   |   |   |   |   |   |   |   |   |   |   |   value=Attribute(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Name(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   id='sys',
|   |   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   |   attr='argv',
|   |   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   slice=Index(
|   |   |   |   |   |   |   |   |   |   |   |   |   value=Num(
|   |   |   |   |   |   |   |   |   |   |   |   |   |   n=0
|   |   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   |   ],
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   op=Add(
|   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   ),
|   |   |   |   |   |   |   |   |   right=Name(
|   |   |   |   |   |   |   |   |   |   id='unparsed',
|   |   |   |   |   |   |   |   |   |   ctx=Load(
|   |   |   |   |   |   |   |   |   |   |   
|   |   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   |   )
|   |   |   |   |   |   |   )
|   |   |   |   |   |   ]
|   |   |   |   |   )
|   |   |   |   )
|   |   |   ],
|   |   |   orelse=[
|   |   |   |   
|   |   |   ]
|   |   )
|   ]
)
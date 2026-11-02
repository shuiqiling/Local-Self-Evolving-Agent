"""Trusted subprocess harness for model-authored pure JSON -> JSON functions.

Linux seccomp blocks file/network/process access before candidate execution.
AST validation adds a deliberately small pure-Python language surface.
"""
import ast
import copy
import ctypes
import errno
import json
import socket
import sys


METHODS={'get','items','keys','values','append','extend','pop','sort','reverse','count','index','add','discard',
         'strip','lstrip','rstrip','split','splitlines','join','replace','lower','upper','startswith','endswith',
         'isdigit','isdecimal','isalnum','copy','update','setdefault'}


def validate(code):
    if not isinstance(code,str) or not 1<=len(code)<=8000:
        raise ValueError('Expected at most8000characters of Python code')
    tree=ast.parse(code)
    if len(list(ast.walk(tree)))>1500:
        raise ValueError('Code too complex')
    denied=(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith,
            ast.AsyncFunctionDef,ast.Lambda)
    for node in ast.walk(tree):
        if isinstance(node,denied):raise ValueError('Unsupported syntax: '+type(node).__name__)
        if isinstance(node,ast.Name) and node.id.startswith('_'):raise ValueError('Private names are unavailable')
        if isinstance(node,ast.Attribute) and node.attr not in METHODS:raise ValueError('Unsupported method: '+node.attr)
        if isinstance(node,ast.FunctionDef) and (node.name.startswith('_') or node.decorator_list):raise ValueError('Unsupported function')
        if isinstance(node,ast.arg) and node.arg.startswith('_'):raise ValueError('Private arguments are unavailable')
    if not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in tree.body):raise ValueError('Define run(arguments)')
    return tree


def restrict():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x7fff0000)
    if not ctx:raise RuntimeError('Cannot initialize seccomp')
    try:
        for name in ['open','openat','openat2','creat','socket','socketpair','connect','bind','listen','accept','accept4',
                     'execve','execveat','fork','vfork','clone','clone3','ptrace','kill','tkill','tgkill',
                     'unlink','unlinkat','rename','renameat','renameat2','mkdir','mkdirat','rmdir','link','linkat',
                     'symlink','symlinkat','mount','umount2','bpf','io_uring_setup','process_vm_readv','process_vm_writev']:
            number=lib.seccomp_syscall_resolve_name(name.encode())
            if number>=0 and lib.seccomp_rule_add(ctx,0x00050000|errno.EPERM,number,0)!=0:raise RuntimeError('seccomp rule failed')
        if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp unavailable; refusing candidate execution')
    finally:lib.seccomp_release(ctx)


def main():
    data=json.load(sys.stdin);tree=validate(data['code']);compiled=compile(tree,'<model-authored-tool>','exec')
    import builtins
    names=['range','len','sum','min','max','abs','sorted','enumerate','list','dict','set','tuple','int','str','float',
           'bool','zip','all','any','map','filter','reversed','isinstance','Exception','ValueError','KeyError','IndexError','TypeError']
    safe={name:getattr(builtins,name) for name in names}
    namespace={'__builtins__':safe}
    restrict()
    if data.get('probe_restrictions'):
        import os
        blocked=False
        try:os.open('/etc/hostname',os.O_RDONLY)
        except PermissionError:blocked=True
        network=False
        try:socket.socket()
        except PermissionError:network=True
        print(json.dumps({'file_access_blocked':blocked,'network_access_blocked':network}));return
    exec(compiled,namespace)
    checks=[]
    for case in data['cases']:
        try:
            value=namespace['run'](copy.deepcopy(case['input']))
            encoded=json.dumps(value,ensure_ascii=False,allow_nan=False)
            if len(encoded)>32000:raise ValueError('Tool output too large')
            entry={'passed':'expected' not in case or json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)==json.dumps(case['expected'],sort_keys=True,ensure_ascii=False,allow_nan=False),'actual':value}
        except Exception as error:entry={'passed':False,'error':type(error).__name__+': '+str(error)[:200]}
        checks.append(entry)
    print(json.dumps({'passed':bool(checks) and all(c['passed'] for c in checks),'cases':checks,'seccomp':True},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:print(json.dumps({'passed':False,'error':type(error).__name__+': '+str(error)[:300]}))

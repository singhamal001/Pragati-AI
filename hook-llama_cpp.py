from cx_Freeze.hooks import collect_all

def hook(finder, module):
    """
    The hook for llama_cpp.
    The collect_all utility automatically finds and includes the package's
    code, data files, and, most importantly, its compiled binaries.
    """
    datas, binaries, hiddenimports = collect_all("llama_cpp")
    finder.include_files(datas)
    finder.include_files(binaries)
    finder.include_hiddenimports(hiddenimports)
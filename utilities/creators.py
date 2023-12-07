from pylmgc90 import pre

def create_materials(dict_mat):
    cont = pre.materials() # container for materials
    for k, v in dict_mat.items():
        mat = pre.material(name=k, **v)
        cont.addMaterial(mat)
    return cont

def create_models(dict_mod):
    mods = pre.models()
    for k, v in dict_mod.items():
        mod = pre.model(name=k, **v)
        mods.addModel(mod)
    return mods

def create_tact_behavs(dict_tact):
    tacts = pre.tact_behavs()
    for k, v in dict_tact.items():
        tact = pre.tact_behav(name=k, **v)
        tacts+=tact
    return tacts

def create_see_tables(dict_see):
    svs = pre.see_tables()
    for k, v in dict_see.items():
        svs+=pre.see_table(**v)
    return svs

def create_postpro_commands(post_dict):
    post = pre.postpro_commands()
    for k, v in post_dict.items():
        post.addCommand(pre.postpro_command(name=k, **v))
    return post
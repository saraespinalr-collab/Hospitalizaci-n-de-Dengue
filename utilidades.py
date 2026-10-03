SINTOMAS = ['cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito']
ALARMA = ['somnolenci', 'hipotensio', 'hepatomeg', 'hem_mucosa', 'hipotermia',
          'aum_hemato', 'caida_plaq', 'acum_liquievento']


def agregar_conteos(X):
    '''Agrega el número de signos de alarma y de síntomas generales presentes.'''
    X = X.copy()
    X['n_alarma'] = X[ALARMA].fillna(0).sum(axis=1)
    X['n_sintomas'] = X[SINTOMAS].fillna(0).sum(axis=1)
    return X

from spexxy.init import InitFromValues
from spexxy.component import Component


class TestFromValues(object):
    def test_default(self):
        # init
        init = InitFromValues({'v': 10, 'Teff': 5800})

        # init mock component
        cmp = Component('star')
        cmp.set('v')
        cmp.set('Teff')

        # run it
        init(cmp, None)

        # compare
        assert 10 == cmp['v']
        assert 5800 == cmp['Teff']

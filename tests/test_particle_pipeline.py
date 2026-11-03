"""V2 particle export and chunked rendering checks; no HPC or real Octavius needed."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest
import warnings
import h5py
import numpy as np
import yaml
from scr import pipeline, input as settings
from scr.particle_output import save_particle_catalogue
from Reader_Object import read_EmissionCatalogue


class TestParticlePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parameters = dict(output={'filename': str(self.root/'particles.hdf5')},
            source={'snapshot_filename':'synthetic', 'catalogue_filename':'synthetic'},
            selection={'target_index':0}, emissivity={'line_labels':['A','B']})
        self.run = dict(redshift=0., unit_factors={},
            frame=dict(center_kpc=[9.5,0,0], bulk_velocity_kms=[1,2,10], box_size_kpc=10., periodic=True,
                       halo_center_kpc=[4.5,0,0], halo_bulk_velocity_kms=[0,0,0]),
            table=dict(table_type='synthetic',table_filename='synthetic',data={'axes':{}}))
        self.gas = dict(snapshot_row=np.array([9,2,5]), particle_id=np.array([2**63+3,7,7],dtype=np.uint64),
            x_kpc=np.array([9.7,.2,4.5]),y_kpc=np.array([0.,.2,0.]),z_kpc=np.zeros(3),
            v_x_kms=np.array([1.,2.,3.]),v_y_kms=np.array([2.,3.,4.]),v_z_kms=np.array([5.,15.,35.]),
            volume_cm3=np.array([2.,3.,5.]), smoothing_length_kpc=np.array([.1,.2,.3]))

    def emissions(self):
        for label, factor in [('A',1.),('B',3.)]:
            yield dict(line_label=label,volume_emissivity_erg_s_cm3=np.full(3,factor),
                       line_luminosity_erg_s=factor*self.gas['volume_cm3'])

    def test_roundtrip_render_conservation_chunks_and_frames(self):
        path=save_particle_catalogue(self.gas,self.emissions(),self.parameters,self.run)
        reader=read_EmissionCatalogue(path)
        np.testing.assert_array_equal(reader.get_fields('particle_id'),self.gas['particle_id'])
        np.testing.assert_array_equal(reader.get_fields('snapshot_row'),[9,2,5])
        for size in (1,2,100):
            image=reader.make_map('A',half_width_kpc=1,bins=4,chunk_size=size)
            self.assertEqual(image['values'].sum(),5.)
            self.assertEqual(image['excluded_luminosity_erg_s'],5.)
            spectrum=reader.make_spectrum('B',[-10,0,10],aperture_radius_kpc=1,chunk_size=size)
            np.testing.assert_array_equal(spectrum['values'],[6,9])
            self.assertEqual(spectrum['excluded_luminosity_erg_s'],15.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,periodic=False)['values'].sum(),2.)
        for los in 'xyz':
            self.assertEqual(reader.make_spectrum('A',[-100,100],los_axis=los)['values'].sum(),10.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,center=[4.5,0,0])['values'].sum(),5.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,center='halo')['values'].sum(),5.)
        np.testing.assert_array_equal(reader.make_spectrum('A',[0,10,20],velocity_frame='halo')['values'],[2.,3.])
        with self.assertRaises(FileExistsError):
            save_particle_catalogue(self.gas,self.emissions(),self.parameters,self.run)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        reader.plot(image); reader.plot(spectrum); plt.close('all')

    def test_failed_and_empty_outputs(self):
        with self.assertRaisesRegex(ValueError,'Missing requested'):
            save_particle_catalogue(self.gas,iter([]),self.parameters,self.run)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.particle_emission_*')))
        empty={name:values[:0] for name,values in self.gas.items()}
        emissions=(dict(line_label=label,volume_emissivity_erg_s_cm3=np.array([]),
                        line_luminosity_erg_s=np.array([])) for label in ('A','B'))
        path=save_particle_catalogue(empty,emissions,self.parameters,self.run)
        self.assertEqual(read_EmissionCatalogue(path).make_map('A')['values'].sum(),0.)

    def test_main_with_real_snapshot_fields_and_table(self):
        snapshot=self.root/'snapshot.hdf5'; table=self.root/'table.hdf5'
        with h5py.File(snapshot,'w') as f:
            f.create_group('Header').attrs.update(Time=.25,HubbleParam=.5,Redshift=3.,BoxSize=10.)
            g=f.create_group('PartType0'); metals=np.zeros((2,11));metals[:,:2]=[.02,.24]
            for name,values in dict(ParticleIDs=np.array([8,8],dtype=np.uint64),
                Coordinates=[[.2,0,0],[.4,0,0]],Velocities=[[2,4,6],[4,6,8]],
                Masses=[1.,2.],Density=[1e-8,2e-8],InternalEnergy=[100.,100.],
                Metallicity=metals,ElectronAbundance=[0.,0.],SmoothingLength=[.1,.2]).items():g[name]=values
        with h5py.File(table,'w') as f:
            f.attrs.update(schema='codex_simba_cloudy_mvp_v1',abundance_model='synthetic')
            f['axes/log_temperature_K']=[0.,12.];f['axes/log_nH_cm3']=[-30.,20.]
            f.create_dataset('lines/labels',data=['A','B'],dtype=h5py.string_dtype())
            d=f.create_dataset('lines/emissivity_erg_s_cm3',data=np.ones((2,2,2))*1e-30)
            d.attrs.update(unit='erg s^-1 cm^-3',axis_order='log_temperature_K, log_nH_cm3, line')
        config=dict(source=dict(snapshot_filename=str(snapshot),catalogue_filename='fake'),
            selection=dict(group_type='halo',target_index=0),
            unit_factors=dict(solar_mass_g=2e33,kpc_to_cm=3e21,specific_energy_to_erg_g=1e10,z_sun_mass_fraction=.02),
            emissivity=dict(type='cloudy_hdf5',table_filename=str(table),line_labels=['B','A']),
            output=dict(filename=str(self.root/'main.hdf5')))
        config_path=self.root/'run.yaml';config_path.write_text(yaml.safe_dump(config))
        cat=MagicMock();cat.__enter__.return_value=cat;cat.redshift=3.;cat.haloes.__len__.return_value=1
        cat.haloes.get_particle_indices.return_value=np.array([1,0])
        cat.haloes.get_dataset.side_effect=lambda name,**kw:dict(r200m=np.array([1.]),
            minpot_pos=np.zeros((1,3)),com_vel_total=np.zeros((1,3)))[name]
        with patch.dict('sys.modules',{'octavius':SimpleNamespace(load_catalogue=lambda filename:cat)}),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            path=pipeline.main(config_path)
        reader=read_EmissionCatalogue(path)
        self.assertEqual(reader.line_labels,('B','A'))
        np.testing.assert_array_equal(reader.get_fields('snapshot_row'),[1,0])
        np.testing.assert_allclose(reader.get_fields('smoothing_length_kpc'),[.1,.05])
        np.testing.assert_allclose(reader.get_fields('position_kpc')[:,0],[.2,.1])
        np.testing.assert_allclose(reader.get_fields('velocity_kms')[:,2],[4.,3.])
        epsilon=reader.get_fields('volume_emissivity_erg_s_cm3',line='A')
        luminosity=reader.get_fields('luminosity_erg_s',line='A')
        np.testing.assert_allclose(epsilon*reader.get_fields('volume_cm3'),luminosity,atol=0)
        with h5py.File(path) as f:self.assertIsNotNone(f['particles/position_kpc'].chunks)
        cat.haloes.get_particle_indices.return_value = np.array([], dtype=np.int64)
        config['output']['filename'] = str(self.root/'empty_main.hdf5')
        config_path.write_text(yaml.safe_dump(config))
        with patch.dict('sys.modules',{'octavius':SimpleNamespace(load_catalogue=lambda filename:cat)}),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            empty = read_EmissionCatalogue(pipeline.main(config_path))
        self.assertEqual(empty.particle_count, 0)
        self.assertEqual(empty.make_spectrum('A', [-10,10])['values'].sum(), 0.)
        config['cube']={'type':'ppp'};config_path.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):settings.read_parameters(config_path)

if __name__=='__main__':unittest.main()

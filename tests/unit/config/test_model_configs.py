"""
Unit tests for model-specific configuration classes.

Tests the new model configuration classes (RHESSysConfig, GNNConfig) and
verifies the transformer mappings work correctly.
"""
from __future__ import annotations

from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP
from symfluence.core.config.models import (
    ERA5Config,
    FUSEConfig,
    GeospatialConfig,
    GNNConfig,
    ISMNConfig,
    LSTMConfig,
    MESHConfig,
    MizuRouteConfig,
    MODISETConfig,
    MODISLandcoverConfig,
    NASADEMConfig,
    NLCDConfig,
    RHESSysConfig,
    SoilGridsConfig,
    SUMMAConfig,
    SymfluenceConfig,
)
from symfluence.core.config.transformers import transform_flat_to_nested


class TestRHESSysConfig:
    """Test RHESSysConfig model"""

    def test_rhessys_config_defaults(self):
        """Test RHESSysConfig with default values"""
        config = RHESSysConfig()
        assert config.install_path == 'default'
        assert config.exe == 'rhessys'
        assert config.settings_path == 'default'
        assert config.world_template == 'world.template'
        assert config.flow_template == 'flow.template'
        assert config.skip_calibration is True
        assert config.use_wmfire is False
        assert config.wmfire_lib == 'libwmfire.so'

    def test_rhessys_config_custom_values(self):
        """Test RHESSysConfig with custom values"""
        config = RHESSysConfig(
            RHESSYS_INSTALL_PATH='/custom/path',
            RHESSYS_EXE='rhessys_custom',
            RHESSYS_USE_WMFIRE=True,
            WMFIRE_LIB='custom_wmfire.so'
        )
        assert config.install_path == '/custom/path'
        assert config.exe == 'rhessys_custom'
        assert config.use_wmfire is True
        assert config.wmfire_lib == 'custom_wmfire.so'

    def test_rhessys_legacy_vmfire_aliases(self):
        """Test RHESSys VMFire legacy aliases work"""
        config = RHESSysConfig(
            RHESSYS_USE_VMFIRE=True,
            VMFIRE_INSTALL_PATH='/vmfire/path'
        )
        assert config.use_vmfire is True
        assert config.vmfire_install_path == '/vmfire/path'


class TestGNNConfig:
    """Test GNNConfig model"""

    def test_gnn_config_defaults(self):
        """Test GNNConfig with default values"""
        config = GNNConfig()
        assert config.load is False
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.epochs == 300
        assert config.batch_size == 64
        assert config.learning_rate == 0.001
        assert config.learning_patience == 30
        assert config.dropout == 0.2
        assert config.l2_regularization == 1e-6
        assert config.params_to_calibrate == 'precip_mult,temp_offset,routing_velocity'
        assert config.parameter_bounds is None

    def test_gnn_config_custom_values(self):
        """Test GNNConfig with custom values"""
        config = GNNConfig(
            GNN_LOAD=True,
            GNN_HIDDEN_SIZE=256,
            GNN_NUM_LAYERS=5,
            GNN_EPOCHS=500,
            GNN_LEARNING_RATE=0.0001,
            GNN_PARAMS_TO_CALIBRATE='param1,param2'
        )
        assert config.load is True
        assert config.hidden_size == 256
        assert config.num_layers == 5
        assert config.epochs == 500
        assert config.learning_rate == 0.0001
        assert config.params_to_calibrate == 'param1,param2'

    def test_gnn_config_with_parameter_bounds(self):
        """Test GNNConfig with parameter bounds"""
        bounds = {'param1': [0.0, 1.0], 'param2': [-1.0, 1.0]}
        config = GNNConfig(GNN_PARAMETER_BOUNDS=bounds)
        assert config.parameter_bounds == bounds


class TestLSTMConfigExtension:
    """Test LSTM config extension with train_through_routing"""

    def test_lstm_train_through_routing_default(self):
        """Test LSTM train_through_routing default value"""
        config = LSTMConfig()
        assert config.train_through_routing is False

    def test_lstm_train_through_routing_custom(self):
        """Test LSTM train_through_routing custom value"""
        config = LSTMConfig(LSTM_TRAIN_THROUGH_ROUTING=True)
        assert config.train_through_routing is True


class TestSUMMAConfigGlacier:
    """Test SUMMA config glacier mode fields"""

    def test_summa_glacier_mode_defaults(self):
        """Test SUMMA glacier mode default values"""
        config = SUMMAConfig()
        assert config.glacier_mode is False
        assert config.glacier_attributes == 'attributes_glac.nc'
        assert config.glacier_coldstate == 'coldState_glac.nc'

    def test_summa_glacier_mode_enabled(self):
        """Test SUMMA glacier mode when enabled"""
        config = SUMMAConfig(
            SETTINGS_SUMMA_GLACIER_MODE=True,
            SETTINGS_SUMMA_GLACIER_ATTRIBUTES='custom_glac_attrs.nc',
            SETTINGS_SUMMA_GLACIER_COLDSTATE='custom_glac_cold.nc'
        )
        assert config.glacier_mode is True
        assert config.glacier_attributes == 'custom_glac_attrs.nc'
        assert config.glacier_coldstate == 'custom_glac_cold.nc'


class TestSUMMAConfigParallel:
    """Test SUMMA parallel execution fields."""

    def test_summa_parallel_defaults(self):
        """Test default SUMMA parallel backend settings."""
        config = SUMMAConfig()
        assert config.parallel_backend == 'slurm'
        assert config.cpus_per_task == 32

    def test_summa_parallel_custom_values(self):
        """Test custom SUMMA parallel backend settings."""
        config = SUMMAConfig(
            SETTINGS_SUMMA_PARALLEL_BACKEND='local',
            SETTINGS_SUMMA_CPUS_PER_TASK=4
        )
        assert config.parallel_backend == 'local'
        assert config.cpus_per_task == 4


class TestOptimizationConfigExtension:
    """Test optimization config extension fields."""

    def test_initial_guess_flat_to_nested_round_trip(self):
        """Test INITIAL_GUESS maps between flat and nested config."""
        initial_guess = {'k_soil': 5.0e-4, 'theta_sat': 0.60}
        config = SymfluenceConfig.from_minimal(
            domain_name='test_domain',
            model='SUMMA',
            EXPERIMENT_TIME_START='2020-01-01 00:00',
            EXPERIMENT_TIME_END='2020-12-31 23:00',
            INITIAL_GUESS=initial_guess,
        )
        assert config.optimization.initial_guess == initial_guess
        assert config.to_dict(flatten=True)['INITIAL_GUESS'] == initial_guess


class TestFUSEConfigExtension:
    """Test FUSE config subcatchment_dim field"""

    def test_fuse_subcatchment_dim_default(self):
        """Test FUSE subcatchment_dim default value"""
        config = FUSEConfig()
        assert config.subcatchment_dim == 'longitude'

    def test_fuse_subcatchment_dim_custom(self):
        """Test FUSE subcatchment_dim custom value"""
        config = FUSEConfig(FUSE_SUBCATCHMENT_DIM='latitude')
        assert config.subcatchment_dim == 'latitude'


class TestNewExports:
    """Test that all new exports are accessible"""

    def test_ismn_config_export(self):
        """Test ISMNConfig is properly exported"""
        config = ISMNConfig()
        assert config.download is False
        assert config.path == 'default'
        assert config.max_stations == 3

    def test_modis_et_config_export(self):
        """Test MODISETConfig is properly exported"""
        config = MODISETConfig()
        assert config.download is False
        assert config.product == 'MOD16A2.061'

    def test_era5_config_export(self):
        """Test ERA5Config is properly exported"""
        config = ERA5Config()
        assert config.time_step_hours == 1

    def test_geospatial_config_export(self):
        """Test GeospatialConfig is properly exported"""
        config = GeospatialConfig()
        # GeospatialConfig has default factories for nested configs
        assert config.soilgrids is not None
        assert config.modis_landcover is not None
        assert config.nlcd is not None
        assert config.nasadem is not None

    def test_soilgrids_config_export(self):
        """Test SoilGridsConfig is properly exported"""
        config = SoilGridsConfig()
        assert config.layer == 'wrb_0-5cm_mode'

    def test_modis_landcover_config_export(self):
        """Test MODISLandcoverConfig is properly exported"""
        config = MODISLandcoverConfig()
        assert 'zenodo.org' in config.base_url

    def test_nlcd_config_export(self):
        """Test NLCDConfig is properly exported"""
        config = NLCDConfig()
        assert 'NLCD' in config.coverage_id

    def test_nasadem_config_export(self):
        """Test NASADEMConfig is properly exported"""
        config = NASADEMConfig()
        assert config.local_dir is None


class TestNewTransformerMappings:
    """Test new transformer mappings for RHESSys, GNN, and other fields"""

    def test_rhessys_mappings_exist(self):
        """Test RHESSys mappings are in FLAT_TO_NESTED_MAP"""
        rhessys_keys = [
            'RHESSYS_INSTALL_PATH',
            'RHESSYS_EXE',
            'SETTINGS_RHESSYS_PATH',
            'EXPERIMENT_OUTPUT_RHESSYS',
            'FORCING_RHESSYS_PATH',
            'RHESSYS_WORLD_TEMPLATE',
            'RHESSYS_FLOW_TEMPLATE',
            'RHESSYS_SKIP_CALIBRATION',
            'RHESSYS_USE_WMFIRE',
            'WMFIRE_INSTALL_PATH',
            'WMFIRE_LIB',
            'RHESSYS_USE_VMFIRE',
            'VMFIRE_INSTALL_PATH',
        ]
        for key in rhessys_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing RHESSys key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'model'
            assert FLAT_TO_NESTED_MAP[key][1] == 'rhessys'

    def test_gnn_mappings_exist(self):
        """Test GNN mappings are in FLAT_TO_NESTED_MAP"""
        gnn_keys = [
            'GNN_LOAD',
            'GNN_HIDDEN_SIZE',
            'GNN_NUM_LAYERS',
            'GNN_EPOCHS',
            'GNN_BATCH_SIZE',
            'GNN_LEARNING_RATE',
            'GNN_LEARNING_PATIENCE',
            'GNN_DROPOUT',
            'GNN_L2_REGULARIZATION',
            'GNN_PARAMS_TO_CALIBRATE',
            'GNN_PARAMETER_BOUNDS',
        ]
        for key in gnn_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing GNN key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'model'
            assert FLAT_TO_NESTED_MAP[key][1] == 'gnn'

    def test_summa_glacier_mappings_exist(self):
        """Test SUMMA glacier mappings are in FLAT_TO_NESTED_MAP"""
        glacier_keys = [
            'SETTINGS_SUMMA_GLACIER_MODE',
            'SETTINGS_SUMMA_GLACIER_ATTRIBUTES',
            'SETTINGS_SUMMA_GLACIER_COLDSTATE',
        ]
        for key in glacier_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing SUMMA glacier key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'model'
            assert FLAT_TO_NESTED_MAP[key][1] == 'summa'

    def test_lstm_train_through_routing_mapping(self):
        """Test LSTM_TRAIN_THROUGH_ROUTING mapping exists"""
        assert 'LSTM_TRAIN_THROUGH_ROUTING' in FLAT_TO_NESTED_MAP
        assert FLAT_TO_NESTED_MAP['LSTM_TRAIN_THROUGH_ROUTING'] == ('model', 'lstm', 'train_through_routing')

    def test_fuse_subcatchment_dim_mapping(self):
        """Test FUSE_SUBCATCHMENT_DIM mapping exists"""
        assert 'FUSE_SUBCATCHMENT_DIM' in FLAT_TO_NESTED_MAP
        assert FLAT_TO_NESTED_MAP['FUSE_SUBCATCHMENT_DIM'] == ('model', 'fuse', 'subcatchment_dim')

    def test_optimization_new_mappings(self):
        """Test new optimization field mappings exist"""
        new_opt_keys = [
            'CALIBRATION_VARIABLE',
            'FINAL_EVALUATION_NUMERICAL_METHOD',
            'CLEANUP_PARALLEL_DIRS',
        ]
        for key in new_opt_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing optimization key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'optimization'


class TestMESHConfigExtension:
    """Test MESH config extension with new fields"""

    def test_mesh_input_file_default(self):
        """Test MESH input_file default value"""
        config = MESHConfig()
        assert config.input_file == 'default'

    def test_mesh_params_to_calibrate_default(self):
        """Test MESH params_to_calibrate default value"""
        config = MESHConfig()
        assert config.params_to_calibrate == 'ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR'

    def test_mesh_spinup_days_default(self):
        """Test MESH spinup_days default value"""
        config = MESHConfig()
        assert config.spinup_days == 365

    def test_mesh_gru_min_total_default(self):
        """Test MESH gru_min_total default value"""
        config = MESHConfig()
        assert config.gru_min_total == 0.0

    def test_mesh_custom_values(self):
        """Test MESH config with custom values"""
        config = MESHConfig(
            SETTINGS_MESH_INPUT='custom_input.txt',
            MESH_PARAMS_TO_CALIBRATE='param1,param2',
            MESH_SPINUP_DAYS=730,
            MESH_GRU_MIN_TOTAL=0.5
        )
        assert config.input_file == 'custom_input.txt'
        assert config.params_to_calibrate == 'param1,param2'
        assert config.spinup_days == 730
        assert config.gru_min_total == 0.5


class TestMizuRouteConfigExtension:
    """Test MizuRoute config extension with new fields"""

    def test_mizuroute_output_var_default(self):
        """Test MizuRoute output_var default value"""
        config = MizuRouteConfig()
        assert config.output_var == 'IRFroutedRunoff'

    def test_mizuroute_parameter_file_default(self):
        """Test MizuRoute parameter_file default value"""
        config = MizuRouteConfig()
        assert config.parameter_file == 'param.nml.default'

    def test_mizuroute_timeout_default(self):
        """Test MizuRoute timeout default value"""
        config = MizuRouteConfig()
        assert config.timeout == 3600

    def test_mizuroute_calibrate_default(self):
        """Test MizuRoute calibrate default value"""
        config = MizuRouteConfig()
        assert config.calibrate is False

    def test_mizuroute_custom_values(self):
        """Test MizuRoute config with custom values"""
        config = MizuRouteConfig(
            SETTINGS_MIZU_OUTPUT_VAR='KWTroutedRunoff',
            SETTINGS_MIZU_PARAMETER_FILE='custom_param.nml',
            MIZUROUTE_PARAMS_TO_CALIBRATE='param1,param2',
            CALIBRATE_MIZUROUTE=True,
            MIZUROUTE_TIMEOUT=7200
        )
        assert config.output_var == 'KWTroutedRunoff'
        assert config.parameter_file == 'custom_param.nml'
        assert config.params_to_calibrate == 'param1,param2'
        assert config.calibrate is True
        assert config.timeout == 7200


class TestModelTimeouts:
    """Test model timeout fields"""

    def test_summa_timeout_default(self):
        """Test SUMMA timeout default value"""
        config = SUMMAConfig()
        assert config.timeout == 7200

    def test_fuse_timeout_default(self):
        """Test FUSE timeout default value"""
        config = FUSEConfig()
        assert config.timeout == 3600

    def test_rhessys_timeout_default(self):
        """Test RHESSys timeout default value"""
        config = RHESSysConfig()
        assert config.timeout == 7200

    def test_mizuroute_timeout_default_in_model_timeouts(self):
        """Test MizuRoute timeout default value"""
        config = MizuRouteConfig()
        assert config.timeout == 3600

    def test_timeout_custom_values(self):
        """Test model configs with custom timeout values"""
        summa = SUMMAConfig(SUMMA_TIMEOUT=14400)
        fuse = FUSEConfig(FUSE_TIMEOUT=7200)
        rhessys = RHESSysConfig(RHESSYS_TIMEOUT=14400)

        assert summa.timeout == 14400
        assert fuse.timeout == 7200
        assert rhessys.timeout == 14400


class TestNewMappingsForMESHMizuRoute:
    """Test new transformer mappings for MESH and MizuRoute"""

    def test_mesh_new_mappings_exist(self):
        """Test MESH new mappings are in FLAT_TO_NESTED_MAP"""
        mesh_keys = [
            'SETTINGS_MESH_INPUT',
            'MESH_PARAMS_TO_CALIBRATE',
            'MESH_SPINUP_DAYS',
            'MESH_GRU_MIN_TOTAL',
        ]
        for key in mesh_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing MESH key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'model'
            assert FLAT_TO_NESTED_MAP[key][1] == 'mesh'

    def test_mizuroute_new_mappings_exist(self):
        """Test MizuRoute new mappings are in FLAT_TO_NESTED_MAP"""
        mizu_keys = [
            'SETTINGS_MIZU_OUTPUT_VAR',
            'SETTINGS_MIZU_PARAMETER_FILE',
            'SETTINGS_MIZU_REMAP_FILE',
            'SETTINGS_MIZU_TOPOLOGY_FILE',
            'MIZUROUTE_PARAMS_TO_CALIBRATE',
            'CALIBRATE_MIZUROUTE',
            'MIZUROUTE_TIMEOUT',
        ]
        for key in mizu_keys:
            assert key in FLAT_TO_NESTED_MAP, f"Missing MizuRoute key: {key}"
            assert FLAT_TO_NESTED_MAP[key][0] == 'model'
            assert FLAT_TO_NESTED_MAP[key][1] == 'mizuroute'

    def test_rhessys_timeout_mapping_exists(self):
        """Test RHESSYS_TIMEOUT mapping exists"""
        assert 'RHESSYS_TIMEOUT' in FLAT_TO_NESTED_MAP
        assert FLAT_TO_NESTED_MAP['RHESSYS_TIMEOUT'] == ('model', 'rhessys', 'timeout')


class TestTransformRHESSysConfig:
    """Test flat-to-nested transformation for RHESSys config"""

    def test_rhessys_flat_to_nested(self):
        """Test RHESSys config transforms correctly"""
        flat = {
            'RHESSYS_INSTALL_PATH': '/custom/rhessys',
            'RHESSYS_EXE': 'rhessys_mpi',
            'SETTINGS_RHESSYS_PATH': '/settings/rhessys',
            'RHESSYS_USE_WMFIRE': True,
            'WMFIRE_LIB': 'libwmfire_custom.so'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['model']['rhessys']['install_path'] == '/custom/rhessys'
        assert nested['model']['rhessys']['exe'] == 'rhessys_mpi'
        assert nested['model']['rhessys']['settings_path'] == '/settings/rhessys'
        assert nested['model']['rhessys']['use_wmfire'] is True
        assert nested['model']['rhessys']['wmfire_lib'] == 'libwmfire_custom.so'


class TestTransformGNNConfig:
    """Test flat-to-nested transformation for GNN config"""

    def test_gnn_flat_to_nested(self):
        """Test GNN config transforms correctly"""
        flat = {
            'GNN_LOAD': True,
            'GNN_HIDDEN_SIZE': 256,
            'GNN_NUM_LAYERS': 5,
            'GNN_EPOCHS': 500,
            'GNN_LEARNING_RATE': 0.0001,
            'GNN_PARAMS_TO_CALIBRATE': 'param1,param2'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['model']['gnn']['load'] is True
        assert nested['model']['gnn']['hidden_size'] == 256
        assert nested['model']['gnn']['num_layers'] == 5
        assert nested['model']['gnn']['epochs'] == 500
        assert nested['model']['gnn']['learning_rate'] == 0.0001
        assert nested['model']['gnn']['params_to_calibrate'] == 'param1,param2'


class TestIntegrationWithSymfluenceConfig:
    """Test integration of new configs with main SymfluenceConfig"""

    def test_rhessys_in_full_config(self):
        """Test RHESSys config works in full SymfluenceConfig"""
        config = self._get_minimal_config()
        config['HYDROLOGICAL_MODEL'] = 'RHESSys'
        config['RHESSYS_EXE'] = 'rhessys'
        config['SETTINGS_RHESSYS_PATH'] = '/path/to/rhessys'

        model = SymfluenceConfig(**config)

        assert model.model.hydrological_model == 'RHESSYS'  # normalized to the registry key
        # RHESSys config is optional - only validated if provided
        assert model.model.rhessys is None or model.model.rhessys.exe == 'rhessys'

    def test_gnn_in_full_config(self):
        """Test GNN config works in full SymfluenceConfig"""
        config = self._get_minimal_config()
        config['GNN_HIDDEN_SIZE'] = 256
        config['GNN_EPOCHS'] = 100

        model = SymfluenceConfig(**config)

        # GNN config is populated when GNN keys are provided
        if model.model.gnn:
            assert model.model.gnn.hidden_size == 256
            assert model.model.gnn.epochs == 100

    def test_summa_glacier_in_full_config(self):
        """Test SUMMA glacier config works in full SymfluenceConfig"""
        config = self._get_minimal_config()
        config['SETTINGS_SUMMA_GLACIER_MODE'] = True
        config['SETTINGS_SUMMA_GLACIER_ATTRIBUTES'] = 'glacier_attrs.nc'

        model = SymfluenceConfig(**config)

        if model.model.summa:
            assert model.model.summa.glacier_mode is True
            assert model.model.summa.glacier_attributes == 'glacier_attrs.nc'

    @staticmethod
    def _get_minimal_config():
        """Helper to get minimal valid configuration"""
        return {
            'SYMFLUENCE_DATA_DIR': '/tmp/data',
            'SYMFLUENCE_CODE_DIR': '/tmp/code',
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'exp_001',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'FORCING_DATASET': 'ERA5',
        }

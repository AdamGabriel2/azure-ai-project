from azureml.core import Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

def deploy_model():
    # Recupera o modelo treinado
    model_path = Model.get_model_path('custom_model')
    model = tf.keras.models.load_model(model_path)

    # Configuração do ambiente de inferência
    environment = Environment(name='inference-env')
    environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['azureml-defaults', 'tensorflow==2.5.0'])

    # Configuração da inferência
    inference_config = InferenceConfig(entry_script='score.py', environment=environment)

    # Implantação usando Azure Container Instances (ACI)
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service = Model.deploy(workspace=run.experiment.workspace,
                           name='custom-model-service',
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deployment_config)
    
    service.wait_for_deployment(show_output=True)
    print(service.state)

if __name__ == '__main__':
    deploy_model()

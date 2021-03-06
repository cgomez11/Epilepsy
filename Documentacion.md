## División de la base de datos: 
1. Leave one out model (name of file: train_data_exp# + train_label_exp#, # is the patient that was left out)\
  -eval: promediar las curvas PR de cada exp  
  -importante para evaluar la dificultad de cada paciente.     
2. 3-fold cross-validation: 3 sets train, val and test, train in two sets and test in one. Execute all possible combinations experiments:  
  --Experiment 1: train w/ train & val; test: test.  
  --Experiment 2: train w/ train & test; test: val.  
  --Experiment 3: train w/ val & test; test: train.  
  Patient distribution in the sets (aleatory + aprox. same number of instances):  
  --Train: 6,21,9,7,4,23,5,8. 24098 instances.  
  --Val: 2,22,20,18,3,10,11,12. 25592 instances.  
  --Test: 16,14,19,17,13,1,24,15. 30156 instances.  

/home/cgomez11/Project/Epilepsy/Dataset/DenseData


## Contenido de archivos en el servidor
1. Dataset: 
/media/user_home2/EEG/Epilepsy/ 
-->Epilepsy/patients Aquí están todos los datos originales de cada paciente (señales crudas).
-->Epilepsy/data
   -->data/DenseData cada 1/8s nueva ventana de crisis (muestreo denso)

- Dentro de chbXX:
chbXX_XX.edf y chbXX_XX.mat (señales originales que se usan para test de registros completos)
  --Para leer los .mat en python usar la función readDenseData.py (/home/cgomez11/Epilepsy)(no es necesario para el entrenamiento, está incluído en la función de entrenar).

- Dentro de chbXX_labels:
chbXX_XX_labels.mat (etiquetas de cada señal 1's y 0's, tiempo en que hay crisis).
Cada 1/4 de segundo hay una etiqueta (64 puntos originales). Para definir si hay crisis se definió un sobrelapamiento de 70% de crisis en esos 64 puntos.  

2. Datos (con muestreo denso) para redes:
/home/cgomez11/Project/Epilepsy/Dataset/DenseData/DataW4_025s_Adj/
- Carpeta DataExp2: partición aleatoria (fija para varios experimentos)
a partir de ordenamiento de pacientes de menor a mayor instancias positivas y negativas. 3 sets: train, val y test. Finalmente se usaron datos raw .mat

- Carpeta dataNN2: .npy de los .mat dentro de DataExp2

- Carpeta NewExpPerPatient: instancias para train y sus respectivas etiquetas para hacer un modelo por paciente. 


3. Códigos manipular señales:
/home/cgomez11/Project/Epilepsy/
- AllTimes_cut.mat: celda de celdas: para cada paciente y cada registro muestra tiempo inicial y final de las crisis. Si no tiene crisis, se asigna 0. 
- regNumber_cut.mat: celda de celdas con nombre de los registros por paciente para cargarlos después. 

- generateDenseImagesInfo.m: genera instancias muestreando denso (parametro shift) en las crisis y guarda un .mat de instancias positivas (celda de 1x24) y .mat de instancias negativas. 

- splitDataExp2_adj.m: hace la partición aleatoria de los pacientes en los 3 sets según número de instancias que tengan. Se remueven outliers por encima y debajo de 3sigmas. Guarda todas las instancias concatenadas en un arreglo 3D y arreglos 3D para los sets de train, val y test con sus respectivas etiquetas (vectores fila). 

- split_data_LOO.m: generar los datos de train para los 24 modelos del Leave-one-out model. Guarda instancias de train y etiquetas.   

- generateDenseInstancesPerPatient.m: genera instancias positivas y negativas de registros de pacientes con más de N_seizures. Por cada paciente guarda un .mat con train_instances y sus etiquetas.

- generateImages2.m: genera instancias de print screen de las señales sin muestrear densamente en las crisis. 

-generateAllRecords.m: generar etiquetas de registros completos de test. 

4. Códigos entrenar y evaluar redes:
/home/cgomez11/Project/Epilepsy/
- network_FC1D.py: red base con 3 fully convolutional. 

- network_FC1D_modX.py: alguna de las diferentes modificaciones probadas sobre la red base. 

- densenet_1D.py: arquitectura de DenseNet en 1D. 

3FOLD CROSS VALIDATION
- train_model_FC1D.py: código para entrenar la red definida en network_FC1D.py (o sus modificaciones). Guarda .pth por cada época y un avg aca y loss de cada época, tanto en train y test. 

- train_model_FC1DExpX.py: lo mismo que la anterior para correr otros exps. 

- train_model_densenet.py: lo mismo que train_model_FC1D.py, pero con la arquitectura definida en densenet_1D.py. 

LOO MODEL
- train_model_LOOEXP.py: código para entrenar cada uno de los 24 modelos con la red network_FC1D.py. Guarda .pth por época y aca y loss en train y test. 

- train_model_LOOEXP_noval.py: como la anterior, pero no hace nada de evaluación. 

- train_model_LOOEXP_noval_densenet.py: lo mismo que train_model_LOOEXP_noval.py pero con la arquitectura de DenseNet. 

EVALUACION:
Todo lo que diga output es que guarda las probabilidades de clase de cada predicción. Los que dicen multi es porque evaluan en 3FCV. 
- eval_main.py:
- eval_main_output.py: prob de clase para LOO
- eval_main_multi.py
- eval_main_multi_output.py: prob de clase para 3FCV

** Los registros que duran más de una hora no se pueden evaluar de una con DenseNet, hay que partirlos (códigos que digan large). 
- eval_main_densenet_large.py:
- eval_main_output_densenet_large.py: prob para LOO
- eval_main_multi_densenet_large.py: 
- eval_main_multi_densenet_large_output.py: prob para 3FCV

5. Resultados Redes:
/home/cgomez11/Project/Epilepsy/Checkpoints/DenseData/DataW4_025s_adj
- Carpeta prueba3: resultados con red original. Dentro hay Exp1, Exp2 y Exp3 que corresponden a cada uno de los folds. Dentro de cada ExpX, hay una carpeta que se llama val, que tiene las predicciones, etiquetas y probabilidades. 

- Carpeta Densnet: igual que prueba3 pero modelos con arq de densenet_FC1D.py. 

- Carpeta LOO: hay 24 carpetas llamadas ExpX, con los .pth de los modelos entrenados con la arquitectura base. 

- Carpeta LOO_res: una carpeta para cada paciente y dentro de c/u hay pred, target y ouput de los registros chbXX. 

- Carpeta LOO_densenet: 24 carpetas ExpXX con los modelos entrenados. 

- Carpeta LOO_densenet_res: carpeta por paciente chbXX con target, pred y output de los registros de test.
 
6. Códigos métricas de evaluación: 
/home/cgomez11/Project/Epilepsy/Checkpoints/DenseData/DataW4_025s_adj/prueba3
- PR_curve_3FCV_python.py: construir curva PR (a partir de predicciones y umbrales) con funciones de python y calcular F-measure y AP. Se calcula para cada paciente en el set de test y se reporta un promedio global (fMax_global) y un txt donde está por paciente. 

/home/cgomez11/Project/Epilepsy/Checkpoints/DenseData/DataW4_025s_adj/LOO_res
- PR_curve_python.py: curva PR con funciones de Python para cada paciente, se extrae Fmax y AP y se promedia el global. 

*Aplica para densenet en 3FCV y LOO, y las modificaciones de la red original que tambien están al nivel de prueba3. 

# Python_Finance

Este es uno de los repositorios en los que muestro los proyectos que he realizado en Python. Se usan tecnicas de Estadistica, Manejo de Datos y Machine Learning. La tematica de la clase mostrada es de finanzas. Se implementan metodos que sirven para la valuacion de acciones.  

En la clase finanzas.py tengo 3 modulos: beta, razones y lstm.

En el modulo beta, implemento un metodo para la obtencion de la beta de una accion. Se utilizan datos obtenido de Yahoo Finance. Se usa el historico del precio en un año de la accion y el historico de IPC.

En el modulo razones, implemento un metodo que regresa las razones financieras de una empresa. Este metodo utiliza informacion recopilada manualmente de la pagina de Yahoo Finance. 

En el modulo lstm (que es el mas importante), implemento un metodo que predice el precio de un accion para n dias en el futuro. Lo hace haciendo uso de una Red Neuronal recurrente LSTM. 

Tambien incluyo algunos archivos csv con el formato adecuado para utilizar lo metodos definidos en las clases. 

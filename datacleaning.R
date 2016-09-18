# Filtrado y limpieza de datos para el proyecto de predicción de resultados académicos 
# Open University Learning Analytics Dataset
# Aprendizaje Automático y Big Data
# Alberto Terceño Ortega - 4º Doble Grado Ingeniería Informática y Matemáticas
# Universidad Complutense de Madrid

library(R.matlab)
library(ade4)


x <- read.csv(file="B:/Documentos/UCM/Erasmus/AABD/proyecto/studentInfo.csv", header=TRUE, sep=",")

#Ejecutar cada una de las líneas comentadas para obtener los datos para otras asignaturas
x <- x[x$code_module == "AAA",]
# x <- x[x$code_module == "BBB",]
# x <- x[x$code_module == "CCC",]

# Levels de los factors del dataset original
# "Equivalente" a age_band <- levels(x$age_band), education <- levels(x$highest_education), regions <- levels(x$region),
# results <- levels(x$final_result)

age_band <- unique(x$age_band)
education <- unique(x$highest_education)
regions <- unique(x$region)
results <- unique(x$final_result)


# Cada level del factor es una columna con indicador 1 o 0
# Los factors que consideramos son género, región, nivel de educación, banda de edad y discapacidad
# En df2 no consideramos el atributo de región
df <- acm.disjonctif(x[,c("gender", "region", "highest_education", "age_band", "disability")])
df2 <- acm.disjonctif(x[,c("gender", "highest_education", "age_band", "disability")])
drops <- c("gender.F", "disability.N")
df <- df[, !(names(df) %in% drops)]
df2 <- df2[, !(names(df2) %in% drops)]

# Añadimos número de intentos, créditos estudiados pero no como factors, sino como integers
# El resultado final será 1 si hay aprobado o matrícula y 0 en caso contrario (suspenso o abandono)
df$attempts <- x[,c("num_of_prev_attempts")]
df$studied_credits <- x[,c("studied_credits")]
df$final_result <- 0
df[x$final_result == "Pass" | x$final_result == "Distinction" , c("final_result")] <- 1

df2$attempts <- x[,c("num_of_prev_attempts")]
df2$studied_credits <- x[,c("studied_credits")]
df2$final_result <- 0
df2[x$final_result == "Pass" | x$final_result == "Distinction" , c("final_result")] <- 1

# Cast a matrices 
a <- as.matrix(df)
b <- as.matrix(df2)

#Guardado de archivos .mat Cambiar nombre para otras asignaturas
writeMat("trainingA.mat", x=a)
writeMat("training2A.mat", x=b)

# Ahora para el resultado final vamos etiquetando según los distintos levels del factor final_result.
df$final_result <- 0
for (i in 1:length(results)){
  df$final_result[x$final_result == results[i]] <- i
}

df2$final_result <- 0
for (i in 1:length(results)){
  df2$final_result[x$final_result == results[i]] <- i
}

# Cast a matrices 
df <- as.matrix(df)
df2 <- as.matrix(df2)

#Guardado de archivos .mat Cambiar nombre para otras asignaturas
writeMat("trainingMulticlassA.mat", x=df)
writeMat("trainingMulticlass2A.mat", x=df2)

La méthode NRC (Network Random Compression) vise à compresser les réseaux de neurones convolutifs (CNN) en exploitant une ensemble ou une base hybride des filtres de convolutions composée de vecteurs ou des filtres de convolutions issus d’une analyse en composantes principales (PCA) et de vecteurs pseudo-aléatoires générés à partir d’un registre à décalage à rétroaction linéaire (Linear Feedback Shift Register, LFSR). 

L’objectif principal est de réduire le nombre de paramètres nécessaires à la représentation des couches convolutives tout en préservant au maximum les performances du réseau. 

La méthode NRC repose sur une factorisation des couches Conv2d dans une base hybride notée BNRC. Cette base est composée de deux types de vecteurs : 
- les vecteurs propres conservés issus de PCA, notés Vs, sont stockés explicitement ; 
- puis certains de ces vecteurs sont remplacés par les vecteurs pseudo- aléatoires, notés Vr, qui peuvent être régénérés à partir d’une graine LFSR de 4 octets. 

Deux hyperparamètres principaux permettent de contrôler le compromis entre la précision du modèle et le gain mémoire : 
- eTh : seuil d’énergie utilisé lors de l’analyse PCA pour déterminer le nombre de dimensions des filtres de convolution s conservées ; 
- p : proportion des remplacement des vecteurs conservées issus de PCA qui seront remplacées par des vecteurs pseudo-aléatoires.
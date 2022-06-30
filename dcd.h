#include <stdio.h>
#include <stdlib.h>

int gdcdp(float *x, float *y, float *z, char file[],long int n,int flag,long int *pos, int wcell)
{
  float dummyf,bfloat[50];
  int N,dummyi,i,imax,headsize;
  long int blsize;
  long int read_dcd_head(FILE *input,int *N,int flag);
  FILE *input;
  
  if(!(input = fopen(file,"rb"))){
    fprintf(stderr,"Could not open file %s\n",file);
    exit(1);
  }

  if(1){
    *pos = read_dcd_head(input,&N,flag);
    //printf("%i\n",*pos);
  }
    //printf("%i\n",*pos);
  rewind(input);
  
  blsize = 6*sizeof(int) + 3*N*sizeof(float) + wcell*(2*sizeof(int)+48);
  //printf("%ld\n",&blsize); 
  fseek(input,(*pos)+n*blsize+wcell*(2*sizeof(int)+48),SEEK_SET);
  
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Reading the x wrong\n");
    fprintf(stderr,"%i != %i\n",dummyi,4*N);
    exit(1);
  }
  fread(x,sizeof(float),N,input);
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Read the x wrong\n");
    exit(1);
  }

  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Reading the y wrong\n");
    exit(1);
  }
  fread(y,sizeof(float),N,input);
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Read the y wrong\n");
   exit(1);
  }

  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Reading the z wrong\n");
    exit(1);
  }
  fread(z,sizeof(float),N,input);
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4*N){
    fprintf(stderr,"Read the z wrong\n");
    exit(1);
  }

  fclose(input);

  return N;
}

long int read_dcd_head(FILE *input, int *N,int flag)
{
  float timestep,dummyf;
  //float TIMEFACTOR=48.88821;
  float TIMEFACTOR=1.0;
  int i,nset,itstart,tbsave,dummyi,nfa,bsize,ntitle;
  char hdr[5],bigbuf[256],title[80],user[100];

  fread(&bsize,sizeof(int),1,input);
  if(bsize != 84){
    i = bsize;
    printf("May be a bad dcd format, but the code is not nescesarly\n");
    printf("portable either.  The value of i should be 84, but it is %i\n",bsize);
  }
  
  fread(hdr,sizeof(char),4,input);
  hdr[4] = '\0';

  fread(&nset,sizeof(int),1,input);
  fread(&itstart,sizeof(int),1,input);
  fread(&tbsave,sizeof(int),1,input);
  fread(&dummyi,sizeof(int),1,input);

  for(i=0;i<5;i++){
    fread(&dummyi,sizeof(int),1,input);
    if(dummyi != 0) fprintf(stderr,"Warning: Not created by NAMD\n");
  }

  fread(&timestep,sizeof(float),1,input);
  fread(&dummyi,sizeof(int),1,input);

  for(i=0;i<8;i++){
    fread(&dummyi,sizeof(int),1,input);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 24){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 84){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 164){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 2){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  
  fread(title,sizeof(char),80,input);
  title[79]='\0';

  fread(user,sizeof(char),80,input);
  user[79]='\0';

  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 164){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(N,sizeof(int),1,input);
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  
  if(flag){
    printf("Number of particles = %i\n",*N);
    printf("Timestep = %f\n",timestep*TIMEFACTOR);
    printf("Number of steps between saves = %i\n",tbsave);
    printf("Starting timestep = %i\n",itstart);
    printf("Number of saved configurations = %i\n",nset);
  }
  
  return ftell(input);
}

long int dcd_info(char file[], int *N,int *nset, int *tbsave, float *timestep, int *wcell)
{
  float dummyf;
  //float TIMEFACTOR=48.88821;
  float TIMEFACTOR = 1.0;
  long int pos;
  int i,itstart,dummyi,nfa,bsize,ntitle;
  char hdr[5],bigbuf[256],title[80],user[100];
  FILE *input;

  if(!(input = fopen(file,"rb"))){
    fprintf(stderr,"Could not open file %s\n",file);
    exit(1);
  }

  fread(&bsize,sizeof(int),1,input);
  if(bsize != 84){
    i = bsize;
    printf("May be a bad dcd format, but the code is not nescesarly\n");
    printf("portable either.  The value of i should be 84, but it is %i\n",i);
  }
  
  fread(hdr,sizeof(char),4,input);
  hdr[4] = '\0';

  fread(nset,sizeof(int),1,input);
  fread(&itstart,sizeof(int),1,input);
  fread(tbsave,sizeof(int),1,input);
  fread(&dummyi,sizeof(int),1,input);

  for(i=0;i<5;i++){
    fread(&dummyi,sizeof(int),1,input);
    if(dummyi != 0) fprintf(stderr,"Warning: Not created by NAMD\n");
  }

  fread(timestep,sizeof(float),1,input);
  fread(wcell,sizeof(int),1,input);
  if(*wcell != 0) *wcell = 1;
  *timestep = TIMEFACTOR*(*timestep);

  for(i=0;i<8;i++){
    fread(&dummyi,sizeof(int),1,input);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 24){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 84){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 164){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 2){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  
  fread(title,sizeof(char),80,input);
  title[79]='\0';

  fread(user,sizeof(char),80,input);
  user[79]='\0';

  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 164){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  fread(N,sizeof(int),1,input);
  fread(&dummyi,sizeof(int),1,input);
  if(dummyi != 4){
    fprintf(stderr,"Error reading dcd header\n");
    exit(1);
  }
  
  pos = ftell(input);
  fclose(input);

  return pos;
}

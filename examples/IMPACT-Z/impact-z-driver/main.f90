!----------------------------------------------------------------
!*** Copyright Notice ***
!IMPACT-Z� Copyright (c) 2016, The Regents of the University of California, through 
!Lawrence Berkeley National Laboratory (subject to receipt of any required approvals 
!from the U.S. Dept. of Energy).  All rights reserved.
!If you have questions about your rights to use or distribute this software, 
!please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.
!NOTICE.  This Software was developed under funding from the U.S. Department of Energy 
!and the U.S. Government consequently retains certain rights. As such, the U.S. Government 
!has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
!worldwide license in the Software to reproduce, distribute copies to the public, prepare 
!derivative works, and perform publicly and display publicly, and to permit other to do so.
!****************************
! Description: v.1.0
! Comments:
!----------------------------------------------------------------
      program main
      use AccSimulatorclass
      implicit none
      include 'mpif.h'
      double precision :: time
      integer ierr,my_rank
	integer nargs,flag, ii, nparm
      real(kind=8), dimension(100)::X
      real(kind=8)::objval=1e30
      character(len=1024)  :: strings
      integer parent

      call MPI_INIT(ierr)
      call MPI_COMM_GET_PARENT(parent, ierr) ! YL: this is needed if this function is spawned by a master process

      call MPI_COMM_RANK(MPI_COMM_WORLD,my_rank,ierr)
      if(my_rank==0)then
            open(unit=3,file='matchquad.in',status='old')
            read(3,*)nparm,ii  
            close(3)
            
            nargs = iargc()
            do ii=1,nparm
            call getarg(ii,strings)
            read(strings,*)X(ii)
            enddo
            call FTN(X, objval,nparm,0)
      endif

      call construct_AccSimulator(time)
      call run_AccSimulator()


      if(my_rank==0)then
            call FTN(X, objval,nparm,1)
            print*,"X:",X(1:nparm), "objective: ",objval
      endif
	if(parent/= MPI_COMM_NULL)then
		call MPI_REDUCE(objval, MPI_BOTTOM, 1, MPI_double_precision, MPI_MIN, 0, parent,ierr)
		call MPI_Comm_disconnect(parent,ierr)
	endif

      call destruct_AccSimulator(time)

      end program main


      subroutine FTN(X, objval,nparm,inout)
            use AccSimulatorclass
            implicit none
            include 'mpif.h'
            integer inout
            integer nparm,ierr,nproc
            real(kind=8), dimension(nparm), intent(in) :: X
            real(kind=8), intent(out) :: objval
            integer:: i
            character*220 :: comst
            character*220 :: strings1,strings2
            character*250 :: comst2
            character, dimension(30,nparm) :: bc
            character :: xbc
            integer, dimension(nparm) :: nline,nrow,nlinein,nrowin
            integer :: j,j1,j2,iline,nlinemax,ii
            integer :: nchars=200,code,j3,j4
            integer :: j5,j6,j7,j8,j9,j50,j4b,j8b,j7b
            character(30) :: tmpstr
            character(30) :: tmpstr2
            real*8,dimension(7) :: fvalue
            real*8,dimension(10) :: objtmp1,objtmp0
            !real*8,dimension(10) :: objtmp1,objtmp0
            real*8 :: fvalueaa,gambet,emtx,emty
            real(kind=8), dimension(nparm) :: XX,x00,xin
            real*8 :: deltx1,deltx2,deltx3,delty1,delty2,delty3
            integer :: nlaser,ipt
            real*8 :: xtmp,xxtmp,x1,x2,x3,x4
            real*8 :: delt1,delt2,delt3,delt4
            integer :: ip,imod
            real*8 :: objtmp01,objtmp03
            integer :: npx,npy,firstflag=1
          

          !read in control parameters and targer values
            open(unit=3,file='matchquad.in',status='old')
            read(3,*)j1,j2  
            read(3,*)nlinein(1:nparm)
            read(3,*)nrowin(1:nparm)
            read(3,*)emtx,emty 
            read(3,*)objtmp0(1:4)
            read(3,*)xtmp
            read(3,*)xin(1:nparm)
            close(3)

          !initial design valudes of the parameters
            x00(1:nparm) = xin(1:nparm)
            nline(1:nparm) = nlinein(1:nparm)
            nrow(1:nparm) = nrowin(1:nparm)
            do i =1, nparm
              XX(i) = x00(i)*(1.0d0+X(i))
            enddo


            if(inout==0)then

                  open(unit=1,file='ImpactZ0.in',status='old')
                  open(unit=2,file='ImpactZ.in',status='unknown')
                  
            ! location of control parameters inside the input file Impactz.in  
            !
            ! quad 1 parameter
            !  nline(1) = 83
            !  nrow(1) = 5
            ! quad 2 parameters
            !  nline(2) = 85
            !  nrow(2) = 5
            ! quad 3 parameters
            !  nline(3) = 87
            !  nrow(3) = 5
            ! quad 4 parameters
            !  nline(4) = 89
            !  nrow(4) = 5
            ! quad 5 parameters
            !  nline(5) = 91
            !  nrow(5) = 5
            !  ...
            !--
            
                  print*,"Xx: ",XX(1:nparm),nparm
                  print*,"nline: ",nline(1:nparm),nparm
                  print*,"nrow: ",nrow(1:nparm),nparm
            !-----------------
            ! prepare Impact input using the control parameters
            !
                  nlinemax = 1000000
                  iline = 0
                  ii = 0
                  do i = 1, nlinemax
                  READ(1, "(A)", ADVANCE="no", IOSTAT=code,end=111) comst
                  
                  ! change the process decomposition if needed here
                  if(comst(1:1)/='!' .and. firstflag==1)then
                        firstflag=0
                        read(comst,*)npx,npy
                        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
                        if(nproc/=npx*npy)then
                        write(*,*)'product of process grids not matching mpi count',nproc,npx,npy
                        
                        npx=2**floor(log(sqrt(dble(nproc)))/log(2d0))
                        npy=nproc/npx
                        if(nproc/=npx*npy)then
                           npx=nproc
                           npy=1
                        endif      
                        write(strings1,*)npx
                        write(strings2,*)npy
                        comst=trim(adjustl(strings1))//" "//trim(adjustl(strings2))
                        ! write(*,*)comst
                        endif
                  endif
            
                  iline = iline + 1
            
                  do ip = 1, nparm
                  if(i.eq.nline(ip)) then
                  j1 = 1
                  j2 = 0
                  j3 = 0
                  j4 = 0
                  j7 = 0
                  j4b = 0
                  j7b = 0
                  do j = 1, nchars
                        if(comst(j:j).ne." ") then
                        j2 = j2 + 1
                        !bc(j2,j1) = comst(j:j)
                        xbc = comst(j:j)
                        else
                        j1 = j1+1
                        j2 = 0
                        endif
                        if(j1.lt.nrow(ip)) then
                        j3 = j3 + 1
                        else if(j1.gt.nrow(ip)) then
                        j4 = j4 + 1
                        else
                        j7 = j7 + 1
                        endif
                  enddo
                  write(tmpstr,*)XX(ip)
                  j8 = len(tmpstr)
                  j8b = 0
                  j4b = 0
            
                  j5 = j3+j8+j4+j8b+j4b
                  do j = 1, j3
                        comst2(j:j) = comst(j:j)
                  enddo
                  do j = j3+1, j3+j8
                        j9 = j - j3
                        comst2(j:j) = tmpstr(j9:j9)
                  enddo
                  do j = j3+j8+1,j3+j8+j4
                        j6 = j-j8+j7
                        comst2(j:j) = comst(j6:j6)
                  enddo
                  write(2,"(A)")comst2(1:j5)
                  imod = 1
                  exit
                  else
                  imod = 0
                  endif
                  !print*,"ip: ",ip,imod
                  enddo
                  if(imod.eq.0)then 
                        write(2,"(A)")trim(comst)
                  endif
                  !print*,"line: ",iline,imod
                  enddo
            111 continue
                  close(1)
                  close(2)
                  
            
            !112 format(A)
            
                  call flush(2)
            else       
            
                  objtmp1 = 0.0d0
                  close(18)
                  close(24)
                  close(25)
                  close(3)
                  !calculate <xz>,<x'z>,<xpz>,<x'pz> from the final particle distribution.
                  open(unit=15,file='fort.18',status='old')
                  do i = 1, nlinemax
                  read(15,*,end=443)fvalue(1:6)
                  enddo
            443 continue
                  close(15)
                  gambet = fvalue(3)*fvalue(5)
            
                  !objtmp0(1) is beta_x,objtmp01 is sig_x
                  objtmp01 = sqrt(objtmp0(1)*emtx/gambet)
                  objtmp03 = sqrt(objtmp0(3)*emty/gambet)
            
                  open(unit=14,file='fort.24',status='old')
                  do i = 1, nlinemax
                  read(14,*,end=444)fvalue(1:7)
                  enddo
            444 continue
                  close(14)
                  objtmp1(1) = fvalue(3)
                  objtmp1(2) = fvalue(6)
            
                  open(unit=13,file='fort.25',status='old')
                  do i = 1, nlinemax
                  read(13,*,end=445)fvalue(1:7)
                  enddo
            445 continue
                  close(13)
                  objtmp1(3) = fvalue(3)
                  objtmp1(4) = fvalue(6)
                  
                  delt1 = (objtmp1(1)-objtmp01)/objtmp01
                  !alpha is dimensionless
                  delt2 = (objtmp1(2)-objtmp0(2))
                  delt3 = (objtmp1(3)-objtmp03)/objtmp03
                  delt4 = (objtmp1(4)-objtmp0(4))
                  
                  objval = sqrt((delt1**2+delt2**2+delt3**2+delt4**2)/4)
            
                  open(unit=11,file='fort.3',status='unknown',position='append')
                  write(11,*)"obj: ",XX(1:nparm),objval
                  close(11)
                  call flush(11)
            endif
          end subroutine FTN
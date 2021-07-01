!----------------------------------------------------------------
! (c) Copyright, 2016 by the Regents of the University of California.
! Inputclass: Input class in I/O module of CONTROL layer. 
! Version: 1.0
! Author: Ji Qiang, LBNL
! Description: This class defines functions to input the global
!              beam and computational parameters and the lattice input
!              parameters in the accelerator.
! Comments: J. Q. modified the source code so that the user can put
!           coments line starting with "!" for each number line in the
!           input file "ImpactZ.in".
!----------------------------------------------------------------
      module Inputclass
#if USE_MPI != 1
        use mpistub
#endif
        interface in_Input
          module procedure in1_Input, in2_Input
        end interface
      contains
        ! Start MPI
        subroutine init_Input(time)
        implicit none
        include 'mpif.h'
        double precision, intent(out) :: time
        integer :: ierr

        ! start MPI library.
        ! call MPI_INIT(ierr)
        time = MPI_WTIME()
        ! for measurement of memory usage.
        !call system_stats()

        end subroutine init_Input
 
        ! Input all parameters except beam line element parameters.
        subroutine in1_Input(odim,onp,onx,ony,onz,oflagbc,oflagdist, &
        orstartflg,oflagmap,distparam,nparam,obcurr,obkenergy,obmass,&
        obcharge,obfreq,oxrad,oyrad,operdlen,onblem,onpcol,onprow,oflagerr,&
        oflagdiag,oflagsbstp,ophsini,onchrg,onptlist,ocurrlist,oqmcclist)

        implicit none
        include 'mpif.h'
        integer, intent(out) :: odim,onp,onx,ony,onz,oflagbc,oflagdist
        integer, intent(out) :: orstartflg,oflagmap,onblem,onpcol,onprow 
        integer, intent(out) :: oflagerr,oflagdiag,oflagsbstp,onchrg
        integer, intent(in) :: nparam
        double precision, dimension(nparam), intent(out) :: distparam
        double precision, intent(out) :: obcurr,obkenergy,obmass
        double precision, intent(out) :: obcharge,obfreq,operdlen,&
                                         oxrad,oyrad,ophsini
        double precision, dimension(:), intent(inout) :: ocurrlist,oqmcclist
        integer, dimension(:), intent(inout) :: onptlist
        double precision :: xjunk
        integer :: my_rank,nproc,ierr,np,itot,njunk1,njunk2,njunk3
        character*1 comst
        integer :: ii,jj,i

        call MPI_COMM_RANK(MPI_COMM_WORLD,my_rank,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,np,ierr)

        if(my_rank.eq.0) then

          print*,"Read input data from file - ImpactZ.in:"
          open(unit=13,file='ImpactZ.in',status='old')

          ii = 0
          jj = 0 
10        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 10
          else
            backspace(13,err=789)
            read(13,*)onpcol,onprow
            ii = ii+1
          endif 
20        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 20
          else
            backspace(13,err=789)
            read(13,*)odim, onp, oflagmap, oflagerr, oflagdiag 
            ii = ii+1
          endif 
30        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 30
          else
            backspace(13,err=789)
            read(13,*)onx, ony, onz, oflagbc, oxrad,oyrad,operdlen
            ii = ii+1
          endif 
40        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 40
          else
            backspace(13,err=789)
            read(13,*)oflagdist, orstartflg, oflagsbstp, onchrg
            ii = ii+1
          endif 
50        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 50
          else
            backspace(13,err=789)
            read(13,*)onptlist(1:onchrg)
            ii = ii+1
          endif 
60        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 60
          else
            backspace(13,err=789)
            read(13,*)ocurrlist(1:onchrg)
            ii = ii+1
          endif 
70        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 70
          else
            backspace(13,err=789)
            read(13,*)oqmcclist(1:onchrg)
            ii = ii+1
          endif 
          distparam = 0.0
80        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 80
          else
            backspace(13,err=789)
            read(13,*)distparam(1),distparam(2),distparam(3),distparam(4),&
                    distparam(5),distparam(6),distparam(7)
            ii = ii+1
          endif 
90        continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 90
          else
            backspace(13,err=789)
            read(13,*)distparam(8),distparam(9),distparam(10),&
                    distparam(11),distparam(12),distparam(13),distparam(14)
            ii = ii+1
          endif
101       continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 101
          else
            backspace(13,err=789)
            read(13,*)distparam(15),distparam(16),distparam(17),&
                    distparam(18),distparam(19),distparam(20),distparam(21)
            ii = ii+1
          endif
102       continue
          read(13,*)comst
          jj = jj + 1
          if(comst.eq."!") then
            goto 102
          else
            backspace(13,err=789)
            read(13,*)obcurr,obkenergy,obmass,obcharge,obfreq,ophsini
            ii = ii+1
          endif

          !count the # of beam line elements.
          itot=0
          njunk3 = 0
123       continue
            read(13,*,end=789)comst
            if(comst.ne."!") then
              backspace(13,err=789)
              read(13,*,end=789)xjunk,njunk1,njunk2,njunk3
              itot = itot + 1
            endif
            if(njunk3.eq.-99)then
              goto 789
            endif
          goto 123
  789     continue
          onblem=itot
!          write(6,*)'onblem = ',onblem
          rewind(13)

          do i = 1, jj
            read(13,*)comst
          enddo
        endif

        call MPI_BCAST(onpcol,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onprow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(odim,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onp,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oflagmap,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oflagerr,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oflagdiag,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onchrg,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onptlist(1),onchrg,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(ocurrlist(1),onchrg,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                       ierr)
        call MPI_BCAST(oqmcclist(1),onchrg,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                       ierr)
        call MPI_BCAST(onx,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(ony,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onz,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oflagbc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oxrad,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(oyrad,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(operdlen,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(oflagdist,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(orstartflg,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(oflagsbstp,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(onblem,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
        call MPI_BCAST(distparam(1),nparam,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                       ierr)
        call MPI_BCAST(obcurr,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(obkenergy,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(obmass,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(obcharge,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(obfreq,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)
        call MPI_BCAST(ophsini,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,&
                         ierr)

        end subroutine in1_Input

        ! Input beam line element parameters.
        subroutine in2_Input(onblem,oblength,obnseg,obmpstp,&
        obtype,value1,value2,value3,value4,value5,value6,value7,value8,&
        value9,value10,value11,value12,value13,value14,value15,value16,&
        value17,value18,value19,value20,value21,value22,value23,value24)
        implicit none
        include 'mpif.h'
        integer,intent(in) :: onblem
        integer,intent(out) :: obnseg(onblem)
        integer,intent(out) :: obmpstp(onblem)
        integer,intent(out) :: obtype(onblem)
        double precision,intent(out) :: oblength(onblem)
        double precision,dimension(onblem),intent(out) :: value1,value2,&
        value3,value4,value5,value6,value7,value8,value9,value10,value11,&
        value12,value13,value14,value15,value16,value17,value18,value19,&
        value20,value21,value22,value23,value24
        integer :: i,irf
        integer :: myrank,ierr
        character*1 comst

        call MPI_COMM_RANK(MPI_COMM_WORLD,myrank,ierr)

        value1 = 0.0
        value2 = 0.0
        value3 = 0.0
        value4 = 0.0
        value5 = 0.0
        value6 = 0.0
        value7 = 0.0
        value8 = 0.0
        value9 = 0.0
        value10 = 0.0
        value11 = 0.0
        value12 = 0.0
        value13 = 0.0
        value14 = 0.0
        value15 = 0.0
        value16 = 0.0
        value17 = 0.0
        value18 = 0.0
        value19 = 0.0
        value20 = 0.0
        value21 = 0.0
        value22 = 0.0
        value23 = 0.0
        value24 = 0.0

        if(myrank.eq.0) then

          i=0
123       continue
            read(13,*,end=789)comst
            if(comst.ne."!") then
              backspace(13,err=789)
              i = i + 1
              read(13,*)oblength(i),obnseg(i),obmpstp(i),obtype(i),&
              value1(i),value2(i),value3(i),value4(i),value5(i),value6(i),&
              value7(i),value8(i),value9(i),value10(i),value11(i),value12(i),&
              value13(i),value14(i),value15(i),value16(i),value17(i),value18(i),&
              value19(i),value20(i),value21(i),value22(i),value23(i),value24(i)
            endif
            if(obtype(i).eq.-99)then
              goto 789
            endif
          goto 123
789       continue

!          print*,"nblem: ",i,onblem
          do i = 1, onblem
!          print*,"value11: ",value11(i),value12(i)
          end do
          close(13)

        endif

        call MPI_BCAST(oblength,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value1,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value2,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value3,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value4,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value5,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value6,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value7,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value8,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value9,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value10,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value11,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value12,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value13,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value14,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value15,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value16,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value17,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value18,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value19,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value20,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value21,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value22,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value23,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(value24,onblem,MPI_DOUBLE_PRECISION,0,&
                       MPI_COMM_WORLD,ierr)
        call MPI_BCAST(obnseg,onblem,MPI_INTEGER,0,MPI_COMM_WORLD,&
                       ierr)
        call MPI_BCAST(obmpstp,onblem,MPI_INTEGER,0,MPI_COMM_WORLD,&
                       ierr)
        call MPI_BCAST(obtype,onblem,MPI_INTEGER,0,MPI_COMM_WORLD,&
                       ierr)

        end subroutine in2_Input

      end module Inputclass


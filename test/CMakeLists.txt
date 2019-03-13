# add_executable( run_vo run_vo.cpp )
# target_link_libraries( run_vo myslam )

add_executable( st sfmtest.cpp )
target_link_libraries( st myslam )

# add_executable( test_lk test_lk.cpp )
# target_link_libraries( test_lk ${OpenCV_LIBS} )
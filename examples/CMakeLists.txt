set(care_depends
    care)

blt_add_executable(NAME StreamLoopExample
                   SOURCES StreamLoop.cpp
                   DEPENDS_ON ${care_depends})

target_include_directories(StreamLoopExample PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(StreamLoopExample PRIVATE ${PROJECT_BINARY_DIR}/include)


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_gl_interop.h>
#include <string>
#include <cmath>

// Shader functions
std::string loadShaderSource(const char* filePath);
GLuint compileShader(GLenum shaderType, const char* filename);
GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath);

// CUDA kernel declaration
extern "C" void launchMandelbrotKernelSurface(cudaArray* textureArray, int width, int height,
                                              float minRe, float maxRe, float minIm, float maxIm,
                                              int maxIterations);

// Global state
struct {
    int width = 800;
    int height = 600;
    float minRe = -2.0f;
    float maxRe = 1.0f;
    float minIm = -1.2f;
    float maxIm = 1.2f;
    double lastX = 0.0;
    double lastY = 0.0;
    bool dragging = false;
} viewState;

// GLFW callbacks
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void windowSizeCallback(GLFWwindow* window, int width, int height);

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "GLFW initialization failed" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(viewState.width, viewState.height, 
                                        "Mandelbrot Explorer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, windowSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW initialization failed" << std::endl;
        return -1;
    }

    // Create shader program
    GLuint shaderProgram = createShaderProgram("/home/omkar/Brendan/Projects/CUDA_Mandelbrot/shaders/mandelbrot.vert", "/home/omkar/Brendan/Projects/CUDA_Mandelbrot/shaders/mandelbrot.frag");
    if (shaderProgram == 0) return -1;
    GLint texUniformLoc = glGetUniformLocation(shaderProgram, "mandelbrotTexture");

    // Fullscreen quad
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // Create texture and CUDA resource
    GLuint texture;
    cudaGraphicsResource* cudaResource = nullptr;
    auto createResources = [&]() {
        // Create/recreate texture
        if(texture) glDeleteTextures(1, &texture);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, viewState.width, viewState.height, 
                    0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Register with CUDA
        if(cudaResource) cudaGraphicsUnregisterResource(cudaResource);
        cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, 
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore);
    };
    
    createResources();  // Initial creation

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // CUDA processing
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaArray* textureArray;
        cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0);

        launchMandelbrotKernelSurface(textureArray, viewState.width, viewState.height,
                                    viewState.minRe, viewState.maxRe, 
                                    viewState.minIm, viewState.maxIm, 1000);

        cudaGraphicsUnmapResources(1, &cudaResource, 0);

        // Rendering
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(texUniformLoc, 0);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &texture);
    glfwTerminate();
    return 0;
}

// Input callbacks implementation
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        viewState.dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &viewState.lastX, &viewState.lastY);
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (viewState.dragging) {
        double dx = xpos - viewState.lastX;
        double dy = ypos - viewState.lastY;
        viewState.lastX = xpos;
        viewState.lastY = ypos;

        float reRange = viewState.maxRe - viewState.minRe;
        float imRange = viewState.maxIm - viewState.minIm;
        
        viewState.minRe -= dx * reRange / viewState.width;
        viewState.maxRe -= dx * reRange / viewState.width;
        viewState.minIm += dy * imRange / viewState.height;
        viewState.maxIm += dy * imRange / viewState.height;
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    float zoomFactor = 1.1f;
    if (yoffset > 0) zoomFactor = 1.0f / zoomFactor;

    // Get mouse position in normalized coordinates
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    float mouseRe = viewState.minRe + (viewState.maxRe - viewState.minRe) * (mouseX / viewState.width);
    float mouseIm = viewState.minIm + (viewState.maxIm - viewState.minIm) * ((viewState.height - mouseY) / viewState.height);

    // Apply zoom
    viewState.minRe = mouseRe + (viewState.minRe - mouseRe) * zoomFactor;
    viewState.maxRe = mouseRe + (viewState.maxRe - mouseRe) * zoomFactor;
    viewState.minIm = mouseIm + (viewState.minIm - mouseIm) * zoomFactor;
    viewState.maxIm = mouseIm + (viewState.maxIm - mouseIm) * zoomFactor;
}

void windowSizeCallback(GLFWwindow* window, int width, int height) {
    viewState.width = width;
    viewState.height = height;
    glViewport(0, 0, width, height);
    
    // Recreate resources with new size
    auto createResources = [&]() {
        // ... (same resource creation as in main)
    };
    createResources();
}

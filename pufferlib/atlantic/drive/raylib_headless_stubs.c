#include "raylib.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *framebuffer = NULL;
static int fb_width = 0;
static int fb_height = 0;
static float line_width = 1.0f;
static Camera3D current_camera = {0};

static void put_pixel(int x, int y, Color color) {
    if (!framebuffer || x < 0 || y < 0 || x >= fb_width || y >= fb_height) {
        return;
    }
    unsigned char *px = framebuffer + 4 * (y * fb_width + x);
    float a = color.a / 255.0f;
    px[0] = (unsigned char)(px[0] * (1.0f - a) + color.r * a);
    px[1] = (unsigned char)(px[1] * (1.0f - a) + color.g * a);
    px[2] = (unsigned char)(px[2] * (1.0f - a) + color.b * a);
    px[3] = 255;
}

static void project_point(Vector3 p, int *x, int *y) {
    float aspect = fb_height > 0 ? (float)fb_width / (float)fb_height : 1.0f;
    float view_h = current_camera.fovy > 0.0f ? current_camera.fovy : 100.0f;
    float view_w = view_h * aspect;
    *x = (int)roundf(((p.x - current_camera.target.x) / view_w + 0.5f) * fb_width);
    *y = (int)roundf(((p.y - current_camera.target.y) / view_h + 0.5f) * fb_height);
}

static void draw_line_pixels(int x0, int y0, int x1, int y1, Color color, float width) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int radius = (int)fmaxf(0.0f, roundf(width * 0.5f));
    while (1) {
        for (int yy = -radius; yy <= radius; yy++) {
            for (int xx = -radius; xx <= radius; xx++) {
                if (xx * xx + yy * yy <= radius * radius + 1) {
                    put_pixel(x0 + xx, y0 + yy, color);
                }
            }
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void draw_disc_pixels(int cx, int cy, int radius, Color color) {
    if (radius < 1) {
        radius = 1;
    }
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x * x + y * y <= radius * radius) {
                put_pixel(cx + x, cy + y, color);
            }
        }
    }
}

static float pixels_per_world_unit(void) {
    float view_h = current_camera.fovy > 0.0f ? current_camera.fovy : 100.0f;
    return fb_height > 0 ? (float)fb_height / view_h : 1.0f;
}

void BeginDrawing(void) {}
void BeginMode3D(Camera3D camera) { current_camera = camera; }
void ClearBackground(Color color) {
    if (!framebuffer) {
        return;
    }
    for (int i = 0; i < fb_width * fb_height; i++) {
        framebuffer[4 * i + 0] = color.r;
        framebuffer[4 * i + 1] = color.g;
        framebuffer[4 * i + 2] = color.b;
        framebuffer[4 * i + 3] = color.a;
    }
}
void CloseWindow(void) {
    free(framebuffer);
    framebuffer = NULL;
    fb_width = 0;
    fb_height = 0;
}
void DrawCircle3D(Vector3 center, float radius, Vector3 rotationAxis, float rotationAngle, Color color) {
    (void)rotationAxis;
    (void)rotationAngle;
    int x, y;
    project_point(center, &x, &y);
    draw_disc_pixels(x, y, (int)roundf(radius * pixels_per_world_unit()), color);
}
void DrawCube(Vector3 position, float width, float height, float length, Color color) {
    (void)height;
    int x, y;
    project_point(position, &x, &y);
    int rx = (int)roundf(width * pixels_per_world_unit() * 0.5f);
    int ry = (int)roundf(length * pixels_per_world_unit() * 0.5f);
    for (int yy = -ry; yy <= ry; yy++) {
        for (int xx = -rx; xx <= rx; xx++) {
            put_pixel(x + xx, y + yy, color);
        }
    }
}
void DrawCubeWires(Vector3 position, float width, float height, float length, Color color) {
    (void)height;
    Vector3 a = {position.x - width * 0.5f, position.y - length * 0.5f, position.z};
    Vector3 b = {position.x + width * 0.5f, position.y - length * 0.5f, position.z};
    Vector3 c = {position.x + width * 0.5f, position.y + length * 0.5f, position.z};
    Vector3 d = {position.x - width * 0.5f, position.y + length * 0.5f, position.z};
    DrawLine3D(a, b, color);
    DrawLine3D(b, c, color);
    DrawLine3D(c, d, color);
    DrawLine3D(d, a, color);
}
void DrawLine3D(Vector3 startPos, Vector3 endPos, Color color) {
    int x0, y0, x1, y1;
    project_point(startPos, &x0, &y0);
    project_point(endPos, &x1, &y1);
    draw_line_pixels(x0, y0, x1, y1, color, line_width);
}
void DrawModelEx(Model model, Vector3 position, Vector3 rotationAxis, float rotationAngle, Vector3 scale, Color tint) {
    (void)model;
    (void)rotationAxis;
    float theta = rotationAngle * (float)M_PI / 180.0f;
    float c = cosf(theta);
    float s = sinf(theta);
    float hx = scale.x * 0.5f;
    float hy = scale.y * 0.5f;
    Vector3 corners[4] = {
        {position.x + hx * c - hy * s, position.y + hx * s + hy * c, position.z},
        {position.x - hx * c - hy * s, position.y - hx * s + hy * c, position.z},
        {position.x - hx * c + hy * s, position.y - hx * s - hy * c, position.z},
        {position.x + hx * c + hy * s, position.y + hx * s - hy * c, position.z},
    };
    for (int i = 0; i < 4; i++) {
        DrawLine3D(corners[i], corners[(i + 1) % 4], tint);
    }
}
void DrawSphere(Vector3 centerPos, float radius, Color color) {
    int x, y;
    project_point(centerPos, &x, &y);
    draw_disc_pixels(x, y, (int)roundf(radius * pixels_per_world_unit()), color);
}
void DrawText(const char *text, int posX, int posY, int fontSize, Color color) {
    (void)text;
    (void)posX;
    (void)posY;
    (void)fontSize;
    (void)color;
}
void DrawTriangle3D(Vector3 v1, Vector3 v2, Vector3 v3, Color color) {
    (void)v1;
    (void)v2;
    (void)v3;
    (void)color;
}
void EndDrawing(void) {}
void EndMode3D(void) {}
Color Fade(Color color, float alpha) {
    color.a = (unsigned char)(alpha * 255.0f);
    return color;
}
BoundingBox GetModelBoundingBox(Model model) {
    (void)model;
    return (BoundingBox){0};
}
Vector2 GetMousePosition(void) { return (Vector2){0}; }
float GetMouseWheelMove(void) { return 0.0f; }
void InitWindow(int width, int height, const char *title) {
    (void)title;
    fb_width = width;
    fb_height = height;
    free(framebuffer);
    framebuffer = calloc((size_t)fb_width * (size_t)fb_height * 4, 1);
}
bool IsKeyDown(int key) {
    (void)key;
    return false;
}
bool IsKeyPressed(int key) {
    (void)key;
    return false;
}
bool IsKeyReleased(int key) {
    (void)key;
    return false;
}
bool IsMouseButtonPressed(int button) {
    (void)button;
    return false;
}
bool IsMouseButtonReleased(int button) {
    (void)button;
    return false;
}
Model LoadModel(const char *fileName) {
    (void)fileName;
    return (Model){0};
}
ModelAnimation *LoadModelAnimations(const char *fileName, int *animCount) {
    (void)fileName;
    if (animCount) {
        *animCount = 0;
    }
    return NULL;
}
int MeasureText(const char *text, int fontSize) {
    (void)fontSize;
    return text ? (int)snprintf(NULL, 0, "%s", text) : 0;
}
void SetConfigFlags(unsigned int flags) { (void)flags; }
void SetTargetFPS(int fps) { (void)fps; }
void SetTraceLogLevel(int logLevel) { (void)logLevel; }
const char *TextFormat(const char *text, ...) {
    static char buffer[1024];
    va_list args;
    va_start(args, text);
    vsnprintf(buffer, sizeof(buffer), text, args);
    va_end(args);
    return buffer;
}
void UnloadModel(Model model) { (void)model; }

void rlPushMatrix(void) {}
void rlPopMatrix(void) {}
void rlTranslatef(float x, float y, float z) {
    (void)x;
    (void)y;
    (void)z;
}
void rlRotatef(float angle, float x, float y, float z) {
    (void)angle;
    (void)x;
    (void)y;
    (void)z;
}
void rlSetLineWidth(float width) { line_width = width; }
unsigned char *rlReadScreenPixels(int width, int height) {
    size_t bytes = (size_t)width * (size_t)height * 4;
    unsigned char *copy = malloc(bytes);
    if (!copy || !framebuffer) {
        free(copy);
        return NULL;
    }
    memcpy(copy, framebuffer, bytes);
    return copy;
}

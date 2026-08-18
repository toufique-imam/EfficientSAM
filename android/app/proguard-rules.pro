# LiteRT loads delegate classes reflectively; stripping them turns a working
# GPU path into a runtime NoClassDefFoundError.
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-dontwarn org.tensorflow.lite.**

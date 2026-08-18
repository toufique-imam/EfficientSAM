plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    // Kotlin 2.0 moved the Compose compiler out of AGP into its own plugin.
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}

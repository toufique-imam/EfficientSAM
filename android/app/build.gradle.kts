plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.example.efficientsam"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.efficientsam"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    androidResources {
        // The interpreter mmaps the model straight out of the APK. A
        // compressed asset cannot be mapped, so it would have to be unpacked
        // to disk first -- and a 95 MB vits encoder makes that painful.
        noCompress += "tflite"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")

    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    // TF Lite runtime + GPU delegate, kept in one family on purpose.
    //
    // The com.google.ai.edge.litert:1.0.1 coordinates look like the modern
    // choice, but litert-gpu:1.0.1 is broken: every GpuDelegate constructor --
    // including the no-arg one -- touches GpuDelegateFactory$Options, and that
    // class ships in none of the litert 1.0.1 artifacts, so building a
    // delegate dies with NoClassDefFoundError at runtime and silently falls
    // back to CPU. Mixing litert with tensorflow-lite-gpu is not an option
    // either: both publish org.tensorflow.lite.* and the build fails on
    // duplicate classes. So use tensorflow-lite for both halves.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")
}

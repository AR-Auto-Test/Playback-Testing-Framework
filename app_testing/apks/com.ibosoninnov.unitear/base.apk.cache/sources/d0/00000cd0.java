package com.google.android.filament;

/* loaded from: classes.dex */
public abstract class Platform {
    private static Platform mCurrentPlatform;

    /* loaded from: classes.dex */
    public static class UnknownPlatform extends Platform {
        private UnknownPlatform() {
        }

        @Override // com.google.android.filament.Platform
        public long getSharedContextNativeHandle(Object obj) {
            return 0L;
        }

        @Override // com.google.android.filament.Platform
        public void log(String str) {
            System.out.println(str);
        }

        @Override // com.google.android.filament.Platform
        public boolean validateSharedContext(Object obj) {
            return false;
        }

        @Override // com.google.android.filament.Platform
        public boolean validateStreamSource(Object obj) {
            return false;
        }

        @Override // com.google.android.filament.Platform
        public boolean validateSurface(Object obj) {
            return false;
        }

        @Override // com.google.android.filament.Platform
        public void warn(String str) {
            System.out.println(str);
        }
    }

    public static Platform get() {
        if (mCurrentPlatform == null) {
            try {
                if (isAndroid()) {
                    mCurrentPlatform = (Platform) Class.forName("com.google.android.filament.AndroidPlatform").newInstance();
                } else {
                    mCurrentPlatform = (Platform) Class.forName("com.google.android.filament.DesktopPlatform").newInstance();
                }
            } catch (Exception unused) {
            }
            if (mCurrentPlatform == null) {
                mCurrentPlatform = new UnknownPlatform();
            }
        }
        return mCurrentPlatform;
    }

    public static boolean isAndroid() {
        return "The Android Project".equalsIgnoreCase(System.getProperty("java.vendor"));
    }

    public static boolean isLinux() {
        return System.getProperty("os.name").contains("Linux") && !isAndroid();
    }

    public static boolean isMacOS() {
        return System.getProperty("os.name").contains("Mac OS X");
    }

    public static boolean isWindows() {
        return System.getProperty("os.name").contains("Windows");
    }

    public abstract long getSharedContextNativeHandle(Object obj);

    public abstract void log(String str);

    public abstract boolean validateSharedContext(Object obj);

    public abstract boolean validateStreamSource(Object obj);

    public abstract boolean validateSurface(Object obj);

    public abstract void warn(String str);
}
package h.a.b;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/* compiled from: ApkLibraryInstaller.java */
/* loaded from: classes2.dex */
public class a implements c {

    /* compiled from: ApkLibraryInstaller.java */
    /* renamed from: h.a.b.a$a  reason: collision with other inner class name */
    /* loaded from: classes2.dex */
    public static class C0129a {

        /* renamed from: a  reason: collision with root package name */
        public ZipFile f6253a;

        /* renamed from: b  reason: collision with root package name */
        public ZipEntry f6254b;

        public C0129a(ZipFile zipFile, ZipEntry zipEntry) {
            this.f6253a = zipFile;
            this.f6254b = zipEntry;
        }
    }

    public final C0129a a(Context context, String[] strArr, String str, f fVar) {
        String[] strArr2;
        ApplicationInfo applicationInfo = context.getApplicationInfo();
        String[] strArr3 = applicationInfo.splitSourceDirs;
        int i = 0;
        if (strArr3 == null || strArr3.length == 0) {
            strArr2 = new String[]{applicationInfo.sourceDir};
        } else {
            strArr2 = new String[strArr3.length + 1];
            strArr2[0] = applicationInfo.sourceDir;
            System.arraycopy(strArr3, 0, strArr2, 1, strArr3.length);
        }
        int length = strArr2.length;
        ZipFile zipFile = null;
        int i2 = 1;
        int i3 = 0;
        while (i < length) {
            String str2 = strArr2[i];
            int i4 = i3;
            while (true) {
                int i5 = i4 + 1;
                if (i4 >= 5) {
                    break;
                }
                try {
                    zipFile = new ZipFile(new File(str2), i2);
                    break;
                } catch (IOException unused) {
                    i4 = i5;
                }
            }
            if (zipFile != null) {
                int i6 = i2;
                int i7 = i3;
                while (true) {
                    int i8 = i3 + 1;
                    if (i3 < 5) {
                        int length2 = strArr.length;
                        int i9 = i6;
                        int i10 = i7;
                        while (i7 < length2) {
                            String str3 = strArr[i7];
                            StringBuilder x = c.b.a.a.a.x("lib");
                            x.append(File.separatorChar);
                            x.append(str3);
                            x.append(File.separatorChar);
                            x.append(str);
                            String sb = x.toString();
                            Object[] objArr = new Object[2];
                            objArr[i10] = sb;
                            objArr[i9] = str2;
                            Objects.requireNonNull(fVar);
                            String.format(Locale.US, "Looking for %s in APK %s...", objArr);
                            ZipEntry entry = zipFile.getEntry(sb);
                            if (entry != null) {
                                return new C0129a(zipFile, entry);
                            }
                            i7++;
                            i10 = 0;
                            i9 = 1;
                        }
                        i7 = i10;
                        i3 = i8;
                        i6 = i9;
                    } else {
                        try {
                            zipFile.close();
                            break;
                        } catch (IOException unused2) {
                        }
                    }
                }
            }
            i++;
            i3 = 0;
            i2 = 1;
        }
        return null;
    }
}
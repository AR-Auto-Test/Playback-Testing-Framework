package h.a.b;

import android.content.Context;
import android.os.Build;
import android.util.Log;
import h.a.b.a;
import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.zip.ZipFile;

/* compiled from: ReLinkerInstance.java */
/* loaded from: classes2.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public final Set<String> f6255a;

    /* renamed from: b  reason: collision with root package name */
    public final d f6256b;

    /* renamed from: c  reason: collision with root package name */
    public final c f6257c;

    /* compiled from: ReLinkerInstance.java */
    /* loaded from: classes2.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Context f6258b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ String f6259c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ String f6260d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ e f6261e;

        public a(Context context, String str, String str2, e eVar) {
            this.f6258b = context;
            this.f6259c = str;
            this.f6260d = str2;
            this.f6261e = eVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            try {
                f.this.d(this.f6258b, this.f6259c, this.f6260d);
                this.f6261e.a();
            } catch (b e2) {
                this.f6261e.b(e2);
            } catch (UnsatisfiedLinkError e3) {
                this.f6261e.b(e3);
            }
        }
    }

    public f() {
        h hVar = new h();
        h.a.b.a aVar = new h.a.b.a();
        this.f6255a = new HashSet();
        this.f6256b = hVar;
        this.f6257c = aVar;
    }

    public File a(Context context) {
        return context.getDir("lib", 0);
    }

    public File b(Context context, String str, String str2) {
        String a2 = ((h) this.f6256b).a(str);
        if (b.v.u.c.m(str2)) {
            return new File(a(context), a2);
        }
        return new File(a(context), c.b.a.a.a.r(a2, ".", str2));
    }

    public void c(Context context, String str, String str2, e eVar) {
        if (context != null) {
            if (!b.v.u.c.m(str)) {
                String.format(Locale.US, "Beginning load of %s...", str);
                if (eVar == null) {
                    d(context, str, str2);
                    return;
                } else {
                    new Thread(new a(context, str, str2, eVar)).start();
                    return;
                }
            }
            throw new IllegalArgumentException("Given library is either null or empty");
        }
        throw new IllegalArgumentException("Given context is null");
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:109:0x015c */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:111:0x0169 */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:78:0x0161 */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:129:0x016b A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:135:0x015e A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:154:0x0171 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:155:0x0171 A[SYNTHETIC] */
    /* JADX WARN: Type inference failed for: r11v2 */
    /* JADX WARN: Type inference failed for: r11v3 */
    /* JADX WARN: Type inference failed for: r11v4 */
    /* JADX WARN: Type inference failed for: r11v5, types: [java.io.Closeable] */
    /* JADX WARN: Type inference failed for: r11v6 */
    /* JADX WARN: Type inference failed for: r11v8, types: [java.io.OutputStream, java.io.FileOutputStream] */
    /* JADX WARN: Type inference failed for: r3v10 */
    /* JADX WARN: Type inference failed for: r3v12 */
    /* JADX WARN: Type inference failed for: r3v8, types: [int, boolean] */
    /* JADX WARN: Type inference failed for: r4v0 */
    /* JADX WARN: Type inference failed for: r4v1, types: [int] */
    /* JADX WARN: Type inference failed for: r4v2 */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void d(Context context, String str, String str2) {
        a.C0129a c0129a;
        a.C0129a a2;
        InputStream inputStream;
        Closeable closeable;
        InputStream inputStream2;
        ?? r11;
        Closeable closeable2;
        boolean z = false;
        int i = 1;
        if (this.f6255a.contains(str)) {
            String.format(Locale.US, "%s already loaded previously!", str);
            return;
        }
        try {
            Objects.requireNonNull((h) this.f6256b);
            System.loadLibrary(str);
            this.f6255a.add(str);
            String.format(Locale.US, "%s (%s) was loaded normally!", str, str2);
        } catch (UnsatisfiedLinkError e2) {
            Object[] objArr = {Log.getStackTraceString(e2)};
            Locale locale = Locale.US;
            String.format(locale, "Loading the library normally failed: %s", objArr);
            String.format(locale, "%s (%s) was not loaded normally, re-linking...", str, str2);
            File b2 = b(context, str, str2);
            if (!b2.exists()) {
                File a3 = a(context);
                File b3 = b(context, str, str2);
                File[] listFiles = a3.listFiles(new g(this, ((h) this.f6256b).a(str)));
                if (listFiles != null) {
                    for (File file : listFiles) {
                        if (!file.getAbsolutePath().equals(b3.getAbsolutePath())) {
                            file.delete();
                        }
                    }
                }
                c cVar = this.f6257c;
                Objects.requireNonNull((h) this.f6256b);
                String[] strArr = Build.SUPPORTED_ABIS;
                if (strArr.length <= 0) {
                    String str3 = Build.CPU_ABI2;
                    strArr = !b.v.u.c.m(str3) ? new String[]{Build.CPU_ABI, str3} : new String[]{Build.CPU_ABI};
                }
                String a4 = ((h) this.f6256b).a(str);
                h.a.b.a aVar = (h.a.b.a) cVar;
                Objects.requireNonNull(aVar);
                try {
                    a2 = aVar.a(context, strArr, a4, this);
                } catch (Throwable th) {
                    th = th;
                    c0129a = null;
                }
                try {
                    if (a2 == null) {
                        throw new b(a4);
                    }
                    int i2 = 0;
                    while (true) {
                        int i3 = i2 + 1;
                        try {
                            if (i2 < 5) {
                                Object[] objArr2 = new Object[i];
                                objArr2[z ? 1 : 0] = a4;
                                String.format(Locale.US, "Found %s! Extracting...", objArr2);
                                try {
                                    if (b2.exists() || b2.createNewFile()) {
                                        try {
                                            inputStream2 = a2.f6253a.getInputStream(a2.f6254b);
                                            try {
                                                r11 = new FileOutputStream(b2);
                                                try {
                                                    byte[] bArr = new byte[4096];
                                                    long j = 0;
                                                    ?? r3 = z;
                                                    boolean z2 = i;
                                                    while (true) {
                                                        int read = inputStream2.read(bArr);
                                                        if (read == -1) {
                                                            break;
                                                        }
                                                        r11.write(bArr, r3, read);
                                                        j += read;
                                                        r3 = 0;
                                                        z2 = true;
                                                    }
                                                    r11.flush();
                                                    r11.getFD().sync();
                                                    if (j != b2.length()) {
                                                        try {
                                                            inputStream2.close();
                                                        } catch (IOException unused) {
                                                        }
                                                        r11.close();
                                                    } else {
                                                        try {
                                                            inputStream2.close();
                                                        } catch (IOException unused2) {
                                                        }
                                                        try {
                                                            r11.close();
                                                        } catch (IOException unused3) {
                                                        }
                                                        b2.setReadable(z2, r3);
                                                        b2.setExecutable(z2, r3);
                                                        b2.setWritable(z2);
                                                        ZipFile zipFile = a2.f6253a;
                                                        if (zipFile != null) {
                                                            zipFile.close();
                                                        }
                                                    }
                                                } catch (FileNotFoundException unused4) {
                                                    if (inputStream2 != null) {
                                                    }
                                                    if (r11 == 0) {
                                                    }
                                                    r11.close();
                                                    i2 = i3;
                                                    z = false;
                                                    i = 1;
                                                } catch (IOException unused5) {
                                                    if (inputStream2 != null) {
                                                    }
                                                    if (r11 == 0) {
                                                    }
                                                    r11.close();
                                                    i2 = i3;
                                                    z = false;
                                                    i = 1;
                                                } catch (Throwable th2) {
                                                    th = th2;
                                                    closeable2 = r11;
                                                    closeable = closeable2;
                                                    inputStream = inputStream2;
                                                    if (inputStream != null) {
                                                        try {
                                                            inputStream.close();
                                                        } catch (IOException unused6) {
                                                        }
                                                    }
                                                    if (closeable != null) {
                                                        try {
                                                            closeable.close();
                                                        } catch (IOException unused7) {
                                                        }
                                                    }
                                                    throw th;
                                                }
                                            } catch (FileNotFoundException unused8) {
                                                r11 = 0;
                                                if (inputStream2 != null) {
                                                    try {
                                                        inputStream2.close();
                                                    } catch (IOException unused9) {
                                                    }
                                                }
                                                if (r11 == 0) {
                                                    i2 = i3;
                                                    z = false;
                                                    i = 1;
                                                }
                                                r11.close();
                                                i2 = i3;
                                                z = false;
                                                i = 1;
                                            } catch (IOException unused10) {
                                                r11 = 0;
                                                if (inputStream2 != null) {
                                                    try {
                                                        inputStream2.close();
                                                    } catch (IOException unused11) {
                                                    }
                                                }
                                                if (r11 == 0) {
                                                    i2 = i3;
                                                    z = false;
                                                    i = 1;
                                                }
                                                r11.close();
                                                i2 = i3;
                                                z = false;
                                                i = 1;
                                            } catch (Throwable th3) {
                                                th = th3;
                                                closeable2 = null;
                                            }
                                        } catch (FileNotFoundException unused12) {
                                            inputStream2 = null;
                                        } catch (IOException unused13) {
                                            inputStream2 = null;
                                        } catch (Throwable th4) {
                                            th = th4;
                                            inputStream = null;
                                            closeable = null;
                                        }
                                    }
                                } catch (IOException unused14) {
                                }
                                i2 = i3;
                                z = false;
                                i = 1;
                            } else {
                                ZipFile zipFile2 = a2.f6253a;
                                if (zipFile2 != null) {
                                    zipFile2.close();
                                }
                            }
                        } catch (IOException unused15) {
                        }
                    }
                } catch (Throwable th5) {
                    th = th5;
                    c0129a = a2;
                    if (c0129a != null) {
                        try {
                            ZipFile zipFile3 = c0129a.f6253a;
                            if (zipFile3 != null) {
                                zipFile3.close();
                            }
                        } catch (IOException unused16) {
                        }
                    }
                    throw th;
                }
            }
            d dVar = this.f6256b;
            String absolutePath = b2.getAbsolutePath();
            Objects.requireNonNull((h) dVar);
            System.load(absolutePath);
            this.f6255a.add(str);
            String.format(Locale.US, "%s (%s) was re-linked!", str, str2);
        }
    }
}
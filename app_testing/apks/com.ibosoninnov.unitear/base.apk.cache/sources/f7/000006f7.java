package c.c.a.k;

import android.annotation.TargetApi;
import android.os.Build;
import android.os.StrictMode;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/* compiled from: DiskLruCache.java */
/* loaded from: classes.dex */
public final class a implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public final File f3460b;

    /* renamed from: c  reason: collision with root package name */
    public final File f3461c;

    /* renamed from: d  reason: collision with root package name */
    public final File f3462d;

    /* renamed from: e  reason: collision with root package name */
    public final File f3463e;

    /* renamed from: f  reason: collision with root package name */
    public final int f3464f;

    /* renamed from: g  reason: collision with root package name */
    public long f3465g;

    /* renamed from: h  reason: collision with root package name */
    public final int f3466h;
    public Writer j;
    public int l;
    public long i = 0;
    public final LinkedHashMap<String, d> k = new LinkedHashMap<>(0, 0.75f, true);
    public long m = 0;
    public final ThreadPoolExecutor n = new ThreadPoolExecutor(0, 1, 60, TimeUnit.SECONDS, new LinkedBlockingQueue(), new b(null));
    public final Callable<Void> o = new CallableC0063a();

    /* compiled from: DiskLruCache.java */
    /* renamed from: c.c.a.k.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class CallableC0063a implements Callable<Void> {
        public CallableC0063a() {
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // java.util.concurrent.Callable
        public Void call() {
            synchronized (a.this) {
                a aVar = a.this;
                if (aVar.j == null) {
                    return null;
                }
                aVar.P();
                if (a.this.I()) {
                    a.this.N();
                    a.this.l = 0;
                }
                return null;
            }
        }
    }

    /* compiled from: DiskLruCache.java */
    /* loaded from: classes.dex */
    public static final class b implements ThreadFactory {
        public b(CallableC0063a callableC0063a) {
        }

        @Override // java.util.concurrent.ThreadFactory
        public synchronized Thread newThread(Runnable runnable) {
            Thread thread;
            thread = new Thread(runnable, "glide-disk-lru-cache-thread");
            thread.setPriority(1);
            return thread;
        }
    }

    /* compiled from: DiskLruCache.java */
    /* loaded from: classes.dex */
    public final class c {

        /* renamed from: a  reason: collision with root package name */
        public final d f3468a;

        /* renamed from: b  reason: collision with root package name */
        public final boolean[] f3469b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f3470c;

        public c(d dVar, CallableC0063a callableC0063a) {
            this.f3468a = dVar;
            this.f3469b = dVar.f3476e ? null : new boolean[a.this.f3466h];
        }

        public void a() {
            a.B(a.this, this, false);
        }

        public File b(int i) {
            File file;
            synchronized (a.this) {
                d dVar = this.f3468a;
                if (dVar.f3477f == this) {
                    if (!dVar.f3476e) {
                        this.f3469b[i] = true;
                    }
                    file = dVar.f3475d[i];
                    a.this.f3460b.mkdirs();
                } else {
                    throw new IllegalStateException();
                }
            }
            return file;
        }
    }

    /* compiled from: DiskLruCache.java */
    /* loaded from: classes.dex */
    public final class d {

        /* renamed from: a  reason: collision with root package name */
        public final String f3472a;

        /* renamed from: b  reason: collision with root package name */
        public final long[] f3473b;

        /* renamed from: c  reason: collision with root package name */
        public File[] f3474c;

        /* renamed from: d  reason: collision with root package name */
        public File[] f3475d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f3476e;

        /* renamed from: f  reason: collision with root package name */
        public c f3477f;

        /* renamed from: g  reason: collision with root package name */
        public long f3478g;

        public d(String str, CallableC0063a callableC0063a) {
            this.f3472a = str;
            int i = a.this.f3466h;
            this.f3473b = new long[i];
            this.f3474c = new File[i];
            this.f3475d = new File[i];
            StringBuilder sb = new StringBuilder(str);
            sb.append('.');
            int length = sb.length();
            for (int i2 = 0; i2 < a.this.f3466h; i2++) {
                sb.append(i2);
                this.f3474c[i2] = new File(a.this.f3460b, sb.toString());
                sb.append(".tmp");
                this.f3475d[i2] = new File(a.this.f3460b, sb.toString());
                sb.setLength(length);
            }
        }

        public String a() {
            long[] jArr;
            StringBuilder sb = new StringBuilder();
            for (long j : this.f3473b) {
                sb.append(' ');
                sb.append(j);
            }
            return sb.toString();
        }

        public final IOException b(String[] strArr) {
            StringBuilder x = c.b.a.a.a.x("unexpected journal line: ");
            x.append(Arrays.toString(strArr));
            throw new IOException(x.toString());
        }
    }

    /* compiled from: DiskLruCache.java */
    /* loaded from: classes.dex */
    public final class e {

        /* renamed from: a  reason: collision with root package name */
        public final File[] f3480a;

        public e(a aVar, String str, long j, File[] fileArr, long[] jArr, CallableC0063a callableC0063a) {
            this.f3480a = fileArr;
        }
    }

    public a(File file, int i, int i2, long j) {
        this.f3460b = file;
        this.f3464f = i;
        this.f3461c = new File(file, "journal");
        this.f3462d = new File(file, "journal.tmp");
        this.f3463e = new File(file, "journal.bkp");
        this.f3466h = i2;
        this.f3465g = j;
    }

    public static void B(a aVar, c cVar, boolean z) {
        synchronized (aVar) {
            d dVar = cVar.f3468a;
            if (dVar.f3477f == cVar) {
                if (z && !dVar.f3476e) {
                    for (int i = 0; i < aVar.f3466h; i++) {
                        if (cVar.f3469b[i]) {
                            if (!dVar.f3475d[i].exists()) {
                                cVar.a();
                                return;
                            }
                        } else {
                            cVar.a();
                            throw new IllegalStateException("Newly created entry didn't create value for index " + i);
                        }
                    }
                }
                for (int i2 = 0; i2 < aVar.f3466h; i2++) {
                    File file = dVar.f3475d[i2];
                    if (z) {
                        if (file.exists()) {
                            File file2 = dVar.f3474c[i2];
                            file.renameTo(file2);
                            long j = dVar.f3473b[i2];
                            long length = file2.length();
                            dVar.f3473b[i2] = length;
                            aVar.i = (aVar.i - j) + length;
                        }
                    } else {
                        E(file);
                    }
                }
                aVar.l++;
                dVar.f3477f = null;
                if (dVar.f3476e | z) {
                    dVar.f3476e = true;
                    aVar.j.append((CharSequence) "CLEAN");
                    aVar.j.append(' ');
                    aVar.j.append((CharSequence) dVar.f3472a);
                    aVar.j.append((CharSequence) dVar.a());
                    aVar.j.append('\n');
                    if (z) {
                        long j2 = aVar.m;
                        aVar.m = 1 + j2;
                        dVar.f3478g = j2;
                    }
                } else {
                    aVar.k.remove(dVar.f3472a);
                    aVar.j.append((CharSequence) "REMOVE");
                    aVar.j.append(' ');
                    aVar.j.append((CharSequence) dVar.f3472a);
                    aVar.j.append('\n');
                }
                G(aVar.j);
                if (aVar.i > aVar.f3465g || aVar.I()) {
                    aVar.n.submit(aVar.o);
                }
                return;
            }
            throw new IllegalStateException();
        }
    }

    @TargetApi(26)
    public static void D(Writer writer) {
        if (Build.VERSION.SDK_INT < 26) {
            writer.close();
            return;
        }
        StrictMode.ThreadPolicy threadPolicy = StrictMode.getThreadPolicy();
        StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder(threadPolicy).permitUnbufferedIo().build());
        try {
            writer.close();
        } finally {
            StrictMode.setThreadPolicy(threadPolicy);
        }
    }

    public static void E(File file) {
        if (file.exists() && !file.delete()) {
            throw new IOException();
        }
    }

    @TargetApi(26)
    public static void G(Writer writer) {
        if (Build.VERSION.SDK_INT < 26) {
            writer.flush();
            return;
        }
        StrictMode.ThreadPolicy threadPolicy = StrictMode.getThreadPolicy();
        StrictMode.setThreadPolicy(new StrictMode.ThreadPolicy.Builder(threadPolicy).permitUnbufferedIo().build());
        try {
            writer.flush();
        } finally {
            StrictMode.setThreadPolicy(threadPolicy);
        }
    }

    public static a J(File file, int i, int i2, long j) {
        if (j > 0) {
            if (i2 > 0) {
                File file2 = new File(file, "journal.bkp");
                if (file2.exists()) {
                    File file3 = new File(file, "journal");
                    if (file3.exists()) {
                        file2.delete();
                    } else {
                        O(file2, file3, false);
                    }
                }
                a aVar = new a(file, i, i2, j);
                if (aVar.f3461c.exists()) {
                    try {
                        aVar.L();
                        aVar.K();
                        return aVar;
                    } catch (IOException e2) {
                        PrintStream printStream = System.out;
                        printStream.println("DiskLruCache " + file + " is corrupt: " + e2.getMessage() + ", removing");
                        aVar.close();
                        c.c.a.k.c.a(aVar.f3460b);
                    }
                }
                file.mkdirs();
                a aVar2 = new a(file, i, i2, j);
                aVar2.N();
                return aVar2;
            }
            throw new IllegalArgumentException("valueCount <= 0");
        }
        throw new IllegalArgumentException("maxSize <= 0");
    }

    public static void O(File file, File file2, boolean z) {
        if (z) {
            E(file2);
        }
        if (!file.renameTo(file2)) {
            throw new IOException();
        }
    }

    public final void C() {
        if (this.j == null) {
            throw new IllegalStateException("cache is closed");
        }
    }

    public c F(String str) {
        synchronized (this) {
            C();
            d dVar = this.k.get(str);
            if (dVar == null) {
                dVar = new d(str, null);
                this.k.put(str, dVar);
            } else if (dVar.f3477f != null) {
                return null;
            }
            c cVar = new c(dVar, null);
            dVar.f3477f = cVar;
            this.j.append((CharSequence) "DIRTY");
            this.j.append(' ');
            this.j.append((CharSequence) str);
            this.j.append('\n');
            G(this.j);
            return cVar;
        }
    }

    public synchronized e H(String str) {
        C();
        d dVar = this.k.get(str);
        if (dVar == null) {
            return null;
        }
        if (dVar.f3476e) {
            for (File file : dVar.f3474c) {
                if (!file.exists()) {
                    return null;
                }
            }
            this.l++;
            this.j.append((CharSequence) "READ");
            this.j.append(' ');
            this.j.append((CharSequence) str);
            this.j.append('\n');
            if (I()) {
                this.n.submit(this.o);
            }
            return new e(this, str, dVar.f3478g, dVar.f3474c, dVar.f3473b, null);
        }
        return null;
    }

    public final boolean I() {
        int i = this.l;
        return i >= 2000 && i >= this.k.size();
    }

    public final void K() {
        E(this.f3462d);
        Iterator<d> it = this.k.values().iterator();
        while (it.hasNext()) {
            d next = it.next();
            int i = 0;
            if (next.f3477f == null) {
                while (i < this.f3466h) {
                    this.i += next.f3473b[i];
                    i++;
                }
            } else {
                next.f3477f = null;
                while (i < this.f3466h) {
                    E(next.f3474c[i]);
                    E(next.f3475d[i]);
                    i++;
                }
                it.remove();
            }
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public final void L() {
        c.c.a.k.b bVar = new c.c.a.k.b(new FileInputStream(this.f3461c), c.c.a.k.c.f3487a);
        try {
            String C = bVar.C();
            String C2 = bVar.C();
            String C3 = bVar.C();
            String C4 = bVar.C();
            String C5 = bVar.C();
            if (!"libcore.io.DiskLruCache".equals(C) || !"1".equals(C2) || !Integer.toString(this.f3464f).equals(C3) || !Integer.toString(this.f3466h).equals(C4) || !"".equals(C5)) {
                throw new IOException("unexpected journal header: [" + C + ", " + C2 + ", " + C4 + ", " + C5 + "]");
            }
            int i = 0;
            while (true) {
                try {
                    M(bVar.C());
                    i++;
                } catch (EOFException unused) {
                    this.l = i - this.k.size();
                    if (bVar.f3485f == -1) {
                        N();
                    } else {
                        this.j = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.f3461c, true), c.c.a.k.c.f3487a));
                    }
                    try {
                        bVar.close();
                        return;
                    } catch (RuntimeException e2) {
                        throw e2;
                    } catch (Exception unused2) {
                        return;
                    }
                }
            }
        } catch (Throwable th) {
            try {
                bVar.close();
            } catch (RuntimeException e3) {
                throw e3;
            } catch (Exception unused3) {
            }
            throw th;
        }
    }

    public final void M(String str) {
        String substring;
        int indexOf = str.indexOf(32);
        if (indexOf != -1) {
            int i = indexOf + 1;
            int indexOf2 = str.indexOf(32, i);
            if (indexOf2 == -1) {
                substring = str.substring(i);
                if (indexOf == 6 && str.startsWith("REMOVE")) {
                    this.k.remove(substring);
                    return;
                }
            } else {
                substring = str.substring(i, indexOf2);
            }
            d dVar = this.k.get(substring);
            if (dVar == null) {
                dVar = new d(substring, null);
                this.k.put(substring, dVar);
            }
            if (indexOf2 != -1 && indexOf == 5 && str.startsWith("CLEAN")) {
                String[] split = str.substring(indexOf2 + 1).split(" ");
                dVar.f3476e = true;
                dVar.f3477f = null;
                if (split.length == a.this.f3466h) {
                    for (int i2 = 0; i2 < split.length; i2++) {
                        try {
                            dVar.f3473b[i2] = Long.parseLong(split[i2]);
                        } catch (NumberFormatException unused) {
                            dVar.b(split);
                            throw null;
                        }
                    }
                    return;
                }
                dVar.b(split);
                throw null;
            } else if (indexOf2 == -1 && indexOf == 5 && str.startsWith("DIRTY")) {
                dVar.f3477f = new c(dVar, null);
                return;
            } else if (indexOf2 != -1 || indexOf != 4 || !str.startsWith("READ")) {
                throw new IOException(c.b.a.a.a.q("unexpected journal line: ", str));
            } else {
                return;
            }
        }
        throw new IOException(c.b.a.a.a.q("unexpected journal line: ", str));
    }

    public final synchronized void N() {
        Writer writer = this.j;
        if (writer != null) {
            D(writer);
        }
        BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.f3462d), c.c.a.k.c.f3487a));
        bufferedWriter.write("libcore.io.DiskLruCache");
        bufferedWriter.write("\n");
        bufferedWriter.write("1");
        bufferedWriter.write("\n");
        bufferedWriter.write(Integer.toString(this.f3464f));
        bufferedWriter.write("\n");
        bufferedWriter.write(Integer.toString(this.f3466h));
        bufferedWriter.write("\n");
        bufferedWriter.write("\n");
        for (d dVar : this.k.values()) {
            if (dVar.f3477f != null) {
                bufferedWriter.write("DIRTY " + dVar.f3472a + '\n');
            } else {
                bufferedWriter.write("CLEAN " + dVar.f3472a + dVar.a() + '\n');
            }
        }
        D(bufferedWriter);
        if (this.f3461c.exists()) {
            O(this.f3461c, this.f3463e, true);
        }
        O(this.f3462d, this.f3461c, false);
        this.f3463e.delete();
        this.j = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.f3461c, true), c.c.a.k.c.f3487a));
    }

    public final void P() {
        while (this.i > this.f3465g) {
            String key = this.k.entrySet().iterator().next().getKey();
            synchronized (this) {
                C();
                d dVar = this.k.get(key);
                if (dVar != null && dVar.f3477f == null) {
                    for (int i = 0; i < this.f3466h; i++) {
                        File file = dVar.f3474c[i];
                        if (file.exists() && !file.delete()) {
                            throw new IOException("failed to delete " + file);
                        }
                        long j = this.i;
                        long[] jArr = dVar.f3473b;
                        this.i = j - jArr[i];
                        jArr[i] = 0;
                    }
                    this.l++;
                    this.j.append((CharSequence) "REMOVE");
                    this.j.append(' ');
                    this.j.append((CharSequence) key);
                    this.j.append('\n');
                    this.k.remove(key);
                    if (I()) {
                        this.n.submit(this.o);
                    }
                }
            }
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public synchronized void close() {
        if (this.j == null) {
            return;
        }
        Iterator it = new ArrayList(this.k.values()).iterator();
        while (it.hasNext()) {
            c cVar = ((d) it.next()).f3477f;
            if (cVar != null) {
                cVar.a();
            }
        }
        P();
        D(this.j);
        this.j = null;
    }
}
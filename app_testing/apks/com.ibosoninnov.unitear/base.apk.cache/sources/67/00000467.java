package b.j.h;

import android.text.SpannableStringBuilder;
import android.text.TextUtils;
import b.j.h.d;
import java.util.Locale;

/* compiled from: BidiFormatter.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static final c f2165a;

    /* renamed from: b  reason: collision with root package name */
    public static final String f2166b;

    /* renamed from: c  reason: collision with root package name */
    public static final String f2167c;

    /* renamed from: d  reason: collision with root package name */
    public static final a f2168d;

    /* renamed from: e  reason: collision with root package name */
    public static final a f2169e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f2170f;

    /* renamed from: g  reason: collision with root package name */
    public final int f2171g;

    /* renamed from: h  reason: collision with root package name */
    public final c f2172h;

    /* compiled from: BidiFormatter.java */
    /* renamed from: b.j.h.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0034a {

        /* renamed from: a  reason: collision with root package name */
        public static final byte[] f2173a = new byte[1792];

        /* renamed from: b  reason: collision with root package name */
        public final CharSequence f2174b;

        /* renamed from: c  reason: collision with root package name */
        public final int f2175c;

        /* renamed from: d  reason: collision with root package name */
        public int f2176d;

        /* renamed from: e  reason: collision with root package name */
        public char f2177e;

        static {
            for (int i = 0; i < 1792; i++) {
                f2173a[i] = Character.getDirectionality(i);
            }
        }

        public C0034a(CharSequence charSequence, boolean z) {
            this.f2174b = charSequence;
            this.f2175c = charSequence.length();
        }

        public byte a() {
            char charAt = this.f2174b.charAt(this.f2176d - 1);
            this.f2177e = charAt;
            if (Character.isLowSurrogate(charAt)) {
                int codePointBefore = Character.codePointBefore(this.f2174b, this.f2176d);
                this.f2176d -= Character.charCount(codePointBefore);
                return Character.getDirectionality(codePointBefore);
            }
            this.f2176d--;
            char c2 = this.f2177e;
            return c2 < 1792 ? f2173a[c2] : Character.getDirectionality(c2);
        }
    }

    static {
        c cVar = d.f2184c;
        f2165a = cVar;
        f2166b = Character.toString((char) 8206);
        f2167c = Character.toString((char) 8207);
        f2168d = new a(false, 2, cVar);
        f2169e = new a(true, 2, cVar);
    }

    public a(boolean z, int i, c cVar) {
        this.f2170f = z;
        this.f2171g = i;
        this.f2172h = cVar;
    }

    /* JADX WARN: Code restructure failed: missing block: B:29:0x0070, code lost:
        if (r3 != 0) goto L48;
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x0073, code lost:
        if (r4 == 0) goto L51;
     */
    /* JADX WARN: Code restructure failed: missing block: B:34:0x0079, code lost:
        if (r0.f2176d <= 0) goto L68;
     */
    /* JADX WARN: Code restructure failed: missing block: B:36:0x007f, code lost:
        switch(r0.a()) {
            case 14: goto L64;
            case 15: goto L64;
            case 16: goto L59;
            case 17: goto L59;
            case 18: goto L55;
            default: goto L67;
        };
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:0x0083, code lost:
        r5 = r5 + 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:39:0x0086, code lost:
        if (r3 != r5) goto L61;
     */
    /* JADX WARN: Code restructure failed: missing block: B:41:0x008a, code lost:
        if (r3 != r5) goto L61;
     */
    /* JADX WARN: Code restructure failed: missing block: B:43:0x008e, code lost:
        r5 = r5 - 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:44:0x0091, code lost:
        return r4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:73:?, code lost:
        return 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:75:?, code lost:
        return 0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:76:?, code lost:
        return 0;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static int a(CharSequence charSequence) {
        byte directionality;
        C0034a c0034a = new C0034a(charSequence, false);
        c0034a.f2176d = 0;
        int i = 0;
        int i2 = 0;
        int i3 = 0;
        while (true) {
            int i4 = c0034a.f2176d;
            if (i4 < c0034a.f2175c && i == 0) {
                char charAt = c0034a.f2174b.charAt(i4);
                c0034a.f2177e = charAt;
                if (Character.isHighSurrogate(charAt)) {
                    int codePointAt = Character.codePointAt(c0034a.f2174b, c0034a.f2176d);
                    c0034a.f2176d = Character.charCount(codePointAt) + c0034a.f2176d;
                    directionality = Character.getDirectionality(codePointAt);
                } else {
                    c0034a.f2176d++;
                    char c2 = c0034a.f2177e;
                    directionality = c2 < 1792 ? C0034a.f2173a[c2] : Character.getDirectionality(c2);
                }
                if (directionality != 0) {
                    if (directionality == 1 || directionality == 2) {
                        if (i3 == 0) {
                        }
                    } else if (directionality != 9) {
                        switch (directionality) {
                            case 14:
                            case 15:
                                i3++;
                                i2 = -1;
                                break;
                            case 16:
                            case 17:
                                i3++;
                                i2 = 1;
                                break;
                            case 18:
                                i3--;
                                i2 = 0;
                                break;
                        }
                    }
                } else if (i3 == 0) {
                }
                i = i3;
            }
        }
        return -1;
    }

    /* JADX WARN: Code restructure failed: missing block: B:30:0x0041, code lost:
        return 1;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static int b(CharSequence charSequence) {
        C0034a c0034a = new C0034a(charSequence, false);
        c0034a.f2176d = c0034a.f2175c;
        int i = 0;
        while (true) {
            int i2 = i;
            while (c0034a.f2176d > 0) {
                byte a2 = c0034a.a();
                if (a2 != 0) {
                    if (a2 == 1 || a2 == 2) {
                        if (i2 != 0) {
                            if (i == 0) {
                                break;
                            }
                        }
                    } else if (a2 != 9) {
                        switch (a2) {
                            case 14:
                            case 15:
                                if (i == i2) {
                                    break;
                                }
                                i2--;
                                break;
                            case 16:
                            case 17:
                                if (i == i2) {
                                    break;
                                }
                                i2--;
                                break;
                            case 18:
                                i2++;
                                break;
                            default:
                                if (i != 0) {
                                    break;
                                } else {
                                    break;
                                }
                                break;
                        }
                        i = i2;
                    } else {
                        continue;
                    }
                } else if (i2 != 0) {
                    if (i == 0) {
                        break;
                    }
                }
            }
            return 0;
        }
        return -1;
    }

    public static a c() {
        Locale locale = Locale.getDefault();
        Locale locale2 = e.f2189a;
        boolean z = TextUtils.getLayoutDirectionFromLocale(locale) == 1;
        c cVar = f2165a;
        if (cVar == f2165a) {
            return z ? f2169e : f2168d;
        }
        return new a(z, 2, cVar);
    }

    public CharSequence d(CharSequence charSequence, c cVar, boolean z) {
        String str;
        if (charSequence == null) {
            return null;
        }
        boolean b2 = ((d.c) cVar).b(charSequence, 0, charSequence.length());
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder();
        String str2 = "";
        if (((this.f2171g & 2) != 0) && z) {
            boolean b3 = ((d.c) (b2 ? d.f2183b : d.f2182a)).b(charSequence, 0, charSequence.length());
            if (!this.f2170f && (b3 || a(charSequence) == 1)) {
                str = f2166b;
            } else {
                str = (!this.f2170f || (b3 && a(charSequence) != -1)) ? "" : f2167c;
            }
            spannableStringBuilder.append((CharSequence) str);
        }
        if (b2 != this.f2170f) {
            spannableStringBuilder.append(b2 ? (char) 8235 : (char) 8234);
            spannableStringBuilder.append(charSequence);
            spannableStringBuilder.append((char) 8236);
        } else {
            spannableStringBuilder.append(charSequence);
        }
        if (z) {
            boolean b4 = ((d.c) (b2 ? d.f2183b : d.f2182a)).b(charSequence, 0, charSequence.length());
            if (!this.f2170f && (b4 || b(charSequence) == 1)) {
                str2 = f2166b;
            } else if (this.f2170f && (!b4 || b(charSequence) == -1)) {
                str2 = f2167c;
            }
            spannableStringBuilder.append((CharSequence) str2);
        }
        return spannableStringBuilder;
    }
}
package b.v;

import android.net.Uri;
import android.os.Bundle;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* compiled from: NavDeepLink.java */
/* loaded from: classes.dex */
public final class h {

    /* renamed from: a  reason: collision with root package name */
    public static final Pattern f2628a = Pattern.compile("^[a-zA-Z]+[+\\w\\-.]*:");

    /* renamed from: b  reason: collision with root package name */
    public final ArrayList<String> f2629b = new ArrayList<>();

    /* renamed from: c  reason: collision with root package name */
    public final Map<String, b> f2630c = new HashMap();

    /* renamed from: d  reason: collision with root package name */
    public Pattern f2631d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f2632e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f2633f;

    /* renamed from: g  reason: collision with root package name */
    public final String f2634g;

    /* renamed from: h  reason: collision with root package name */
    public Pattern f2635h;
    public final String i;

    /* compiled from: NavDeepLink.java */
    /* loaded from: classes.dex */
    public static class a implements Comparable<a> {

        /* renamed from: b  reason: collision with root package name */
        public String f2636b;

        /* renamed from: c  reason: collision with root package name */
        public String f2637c;

        public a(String str) {
            String[] split = str.split("/", -1);
            this.f2636b = split[0];
            this.f2637c = split[1];
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // java.lang.Comparable
        /* renamed from: a */
        public int compareTo(a aVar) {
            int i = this.f2636b.equals(aVar.f2636b) ? 2 : 0;
            return this.f2637c.equals(aVar.f2637c) ? i + 1 : i;
        }
    }

    /* compiled from: NavDeepLink.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public String f2638a;

        /* renamed from: b  reason: collision with root package name */
        public ArrayList<String> f2639b = new ArrayList<>();
    }

    public h(String str, String str2, String str3) {
        this.f2631d = null;
        int i = 0;
        this.f2632e = false;
        this.f2633f = false;
        this.f2635h = null;
        this.f2634g = str2;
        this.i = str3;
        if (str != null) {
            Uri parse = Uri.parse(str);
            int i2 = 1;
            this.f2633f = parse.getQuery() != null;
            StringBuilder sb = new StringBuilder("^");
            if (!f2628a.matcher(str).find()) {
                sb.append("http[s]?://");
            }
            Pattern compile = Pattern.compile("\\{(.+?)\\}");
            if (this.f2633f) {
                Matcher matcher = Pattern.compile("(\\?)").matcher(str);
                if (matcher.find()) {
                    a(str.substring(0, matcher.start()), sb, compile);
                }
                this.f2632e = false;
                for (String str4 : parse.getQueryParameterNames()) {
                    StringBuilder sb2 = new StringBuilder();
                    String queryParameter = parse.getQueryParameter(str4);
                    Matcher matcher2 = compile.matcher(queryParameter);
                    b bVar = new b();
                    while (matcher2.find()) {
                        bVar.f2639b.add(matcher2.group(i2));
                        sb2.append(Pattern.quote(queryParameter.substring(i, matcher2.start())));
                        sb2.append("(.+?)?");
                        i = matcher2.end();
                        i2 = 1;
                    }
                    if (i < queryParameter.length()) {
                        sb2.append(Pattern.quote(queryParameter.substring(i)));
                    }
                    bVar.f2638a = sb2.toString().replace(".*", "\\E.*\\Q");
                    this.f2630c.put(str4, bVar);
                    i = 0;
                    i2 = 1;
                }
            } else {
                this.f2632e = a(str, sb, compile);
            }
            this.f2631d = Pattern.compile(sb.toString().replace(".*", "\\E.*\\Q"), 2);
        }
        if (str3 != null) {
            if (Pattern.compile("^[\\s\\S]+/[\\s\\S]+$").matcher(str3).matches()) {
                a aVar = new a(str3);
                StringBuilder x = c.b.a.a.a.x("^(");
                x.append(aVar.f2636b);
                x.append("|[*]+)/(");
                x.append(aVar.f2637c);
                x.append("|[*]+)$");
                this.f2635h = Pattern.compile(x.toString().replace("*|[*]", "[\\s\\S]"));
                return;
            }
            throw new IllegalArgumentException(c.b.a.a.a.r("The given mimeType ", str3, " does not match to required \"type/subtype\" format"));
        }
    }

    public final boolean a(String str, StringBuilder sb, Pattern pattern) {
        Matcher matcher = pattern.matcher(str);
        boolean z = !str.contains(".*");
        int i = 0;
        while (matcher.find()) {
            this.f2629b.add(matcher.group(1));
            sb.append(Pattern.quote(str.substring(i, matcher.start())));
            sb.append("(.+?)");
            i = matcher.end();
            z = false;
        }
        if (i < str.length()) {
            sb.append(Pattern.quote(str.substring(i)));
        }
        sb.append("($|(\\?(.)*))");
        return z;
    }

    public final boolean b(Bundle bundle, String str, String str2, d dVar) {
        if (dVar != null) {
            p pVar = dVar.f2611a;
            try {
                pVar.d(bundle, str, pVar.c(str2));
                return false;
            } catch (IllegalArgumentException unused) {
                return true;
            }
        }
        bundle.putString(str, str2);
        return false;
    }
}
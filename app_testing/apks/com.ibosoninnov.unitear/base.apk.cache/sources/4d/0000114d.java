package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzat implements Iterable, zzap {
    private final String zza;

    public zzat(String str) {
        if (str == null) {
            throw new IllegalArgumentException("StringValue cannot be null.");
        }
        this.zza = str;
    }

    public final boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof zzat) {
            return this.zza.equals(((zzat) obj).zza);
        }
        return false;
    }

    public final int hashCode() {
        return this.zza.hashCode();
    }

    @Override // java.lang.Iterable
    public final Iterator iterator() {
        return new zzas(this);
    }

    public final String toString() {
        return a.r("\"", this.zza, "\"");
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:245:0x0306 */
    /* JADX DEBUG: Multi-variable search result rejected for r3v84, resolved type: int */
    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Code restructure failed: missing block: B:136:0x02fa, code lost:
        if (r1[r7].isEmpty() == false) goto L99;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:101:0x0166  */
    /* JADX WARN: Removed duplicated region for block: B:103:0x0170  */
    /* JADX WARN: Removed duplicated region for block: B:104:0x0185  */
    /* JADX WARN: Removed duplicated region for block: B:105:0x019c  */
    /* JADX WARN: Removed duplicated region for block: B:106:0x01a7  */
    /* JADX WARN: Removed duplicated region for block: B:107:0x01be  */
    /* JADX WARN: Removed duplicated region for block: B:108:0x01d4  */
    /* JADX WARN: Removed duplicated region for block: B:109:0x01e9  */
    /* JADX WARN: Removed duplicated region for block: B:119:0x0267  */
    /* JADX WARN: Removed duplicated region for block: B:146:0x031c  */
    /* JADX WARN: Removed duplicated region for block: B:163:0x03b3  */
    /* JADX WARN: Removed duplicated region for block: B:170:0x03ff  */
    /* JADX WARN: Removed duplicated region for block: B:182:0x0482  */
    /* JADX WARN: Removed duplicated region for block: B:190:0x04d2  */
    /* JADX WARN: Removed duplicated region for block: B:203:0x0530  */
    /* JADX WARN: Removed duplicated region for block: B:212:0x0585  */
    /* JADX WARN: Removed duplicated region for block: B:223:0x05cd  */
    /* JADX WARN: Removed duplicated region for block: B:231:0x0608  */
    /* JADX WARN: Removed duplicated region for block: B:43:0x00b6  */
    /* JADX WARN: Removed duplicated region for block: B:44:0x00ba  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x00c3  */
    /* JADX WARN: Removed duplicated region for block: B:50:0x00cc  */
    /* JADX WARN: Removed duplicated region for block: B:53:0x00d5  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x00de  */
    /* JADX WARN: Removed duplicated region for block: B:59:0x00e7  */
    /* JADX WARN: Removed duplicated region for block: B:62:0x00ef  */
    /* JADX WARN: Removed duplicated region for block: B:65:0x00fa  */
    /* JADX WARN: Removed duplicated region for block: B:68:0x0103  */
    /* JADX WARN: Removed duplicated region for block: B:71:0x010b  */
    /* JADX WARN: Removed duplicated region for block: B:74:0x0114  */
    /* JADX WARN: Removed duplicated region for block: B:77:0x011c  */
    /* JADX WARN: Removed duplicated region for block: B:80:0x0125  */
    /* JADX WARN: Removed duplicated region for block: B:84:0x012f  */
    /* JADX WARN: Removed duplicated region for block: B:87:0x0139  */
    /* JADX WARN: Removed duplicated region for block: B:90:0x0144  */
    /* JADX WARN: Removed duplicated region for block: B:95:0x0154  */
    @Override // com.google.android.gms.internal.measurement.zzap
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final zzap zzbR(String str, zzg zzgVar, List list) {
        String str2;
        String str3;
        String str4;
        String str5;
        char c2;
        String str6;
        zzap zzatVar;
        zzat zzatVar2;
        int i;
        double doubleValue;
        zzap zzahVar;
        Matcher matcher;
        double min;
        double min2;
        zzat zzatVar3;
        int i2;
        int i3;
        int i4;
        zzg zzgVar2;
        int i5;
        int length;
        if ("charAt".equals(str) || "concat".equals(str) || "hasOwnProperty".equals(str) || "indexOf".equals(str) || "lastIndexOf".equals(str) || "match".equals(str) || "replace".equals(str) || FirebaseAnalytics.Event.SEARCH.equals(str) || "slice".equals(str) || "split".equals(str) || "substring".equals(str) || "toLowerCase".equals(str) || "toLocaleLowerCase".equals(str) || "toString".equals(str) || "toUpperCase".equals(str)) {
            str2 = "toLocaleUpperCase";
        } else {
            str2 = "toLocaleUpperCase";
            if (!str2.equals(str)) {
                str3 = "hasOwnProperty";
                if (!"trim".equals(str)) {
                    throw new IllegalArgumentException(String.format("%s is not a String function", str));
                }
                switch (str.hashCode()) {
                    case -1789698943:
                        str4 = "charAt";
                        str5 = str3;
                        if (str.equals(str5)) {
                            c2 = 2;
                            break;
                        }
                        c2 = 65535;
                        break;
                    case -1776922004:
                        String str7 = "charAt";
                        str4 = str7;
                        if (str.equals("toString")) {
                            c2 = 14;
                            str4 = str7;
                            str5 = str3;
                            break;
                        }
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -1464939364:
                        String str8 = "charAt";
                        str4 = str8;
                        if (str.equals("toLocaleLowerCase")) {
                            c2 = '\f';
                            str4 = str8;
                            str5 = str3;
                            break;
                        }
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -1361633751:
                        String str9 = "charAt";
                        boolean equals = str.equals(str9);
                        str4 = str9;
                        if (equals) {
                            c2 = 0;
                            str4 = str9;
                            str5 = str3;
                            break;
                        }
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -1354795244:
                        if (str.equals("concat")) {
                            c2 = 1;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -1137582698:
                        if (str.equals("toLowerCase")) {
                            c2 = '\r';
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -906336856:
                        if (str.equals(FirebaseAnalytics.Event.SEARCH)) {
                            c2 = 7;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -726908483:
                        if (str.equals(str2)) {
                            c2 = 11;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -467511597:
                        if (str.equals("lastIndexOf")) {
                            c2 = 4;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case -399551817:
                        if (str.equals("toUpperCase")) {
                            c2 = 15;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 3568674:
                        if (str.equals("trim")) {
                            c2 = 16;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 103668165:
                        if (str.equals("match")) {
                            c2 = 5;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 109526418:
                        if (str.equals("slice")) {
                            c2 = '\b';
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 109648666:
                        if (str.equals("split")) {
                            c2 = '\t';
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 530542161:
                        if (str.equals("substring")) {
                            c2 = '\n';
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 1094496948:
                        if (str.equals("replace")) {
                            c2 = 6;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    case 1943291465:
                        if (str.equals("indexOf")) {
                            c2 = 3;
                            str4 = "charAt";
                            str5 = str3;
                            break;
                        }
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                    default:
                        str4 = "charAt";
                        str5 = str3;
                        c2 = 65535;
                        break;
                }
                str6 = "undefined";
                switch (c2) {
                    case 0:
                        zzh.zzj(str4, 1, list);
                        int zza = !list.isEmpty() ? (int) zzh.zza(zzgVar.zzb((zzap) list.get(0)).zzh().doubleValue()) : 0;
                        String str10 = this.zza;
                        if (zza >= 0 && zza < str10.length()) {
                            zzatVar = new zzat(String.valueOf(str10.charAt(zza)));
                            return zzatVar;
                        }
                        return zzap.zzm;
                    case 1:
                        zzatVar2 = this;
                        if (!list.isEmpty()) {
                            StringBuilder sb = new StringBuilder(zzatVar2.zza);
                            for (int i6 = 0; i6 < list.size(); i6++) {
                                sb.append(zzgVar.zzb((zzap) list.get(i6)).zzi());
                            }
                            return new zzat(sb.toString());
                        }
                        return zzatVar2;
                    case 2:
                        zzh.zzh(str5, 1, list);
                        String str11 = this.zza;
                        zzap zzb = zzgVar.zzb((zzap) list.get(0));
                        if ("length".equals(zzb.zzi())) {
                            return zzap.zzk;
                        }
                        double doubleValue2 = zzb.zzh().doubleValue();
                        return (doubleValue2 != Math.floor(doubleValue2) || (i = (int) doubleValue2) < 0 || i >= str11.length()) ? zzap.zzl : zzap.zzk;
                    case 3:
                        double d2 = ShadowDrawableWrapper.COS_45;
                        zzh.zzj("indexOf", 2, list);
                        String str12 = this.zza;
                        String zzi = list.size() > 0 ? zzgVar.zzb((zzap) list.get(0)).zzi() : "undefined";
                        if (list.size() >= 2) {
                            d2 = zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue();
                        }
                        zzatVar = new zzah(Double.valueOf(str12.indexOf(zzi, (int) zzh.zza(d2))));
                        return zzatVar;
                    case 4:
                        zzh.zzj("lastIndexOf", 2, list);
                        String str13 = this.zza;
                        String zzi2 = list.size() > 0 ? zzgVar.zzb((zzap) list.get(0)).zzi() : "undefined";
                        zzahVar = new zzah(Double.valueOf(str13.lastIndexOf(zzi2, (int) (Double.isNaN(list.size() < 2 ? Double.NaN : zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()) ? Double.POSITIVE_INFINITY : zzh.zza(doubleValue)))));
                        return zzahVar;
                    case 5:
                        zzh.zzj("match", 1, list);
                        Matcher matcher2 = Pattern.compile(list.size() <= 0 ? "" : zzgVar.zzb((zzap) list.get(0)).zzi()).matcher(this.zza);
                        return matcher2.find() ? new zzae(Arrays.asList(new zzat(matcher2.group()))) : zzap.zzg;
                    case 6:
                        zzatVar2 = this;
                        zzh.zzj("replace", 2, list);
                        zzap zzapVar = zzap.zzf;
                        if (!list.isEmpty()) {
                            str6 = zzgVar.zzb((zzap) list.get(0)).zzi();
                            if (list.size() > 1) {
                                zzapVar = zzgVar.zzb((zzap) list.get(1));
                            }
                        }
                        String str14 = str6;
                        String str15 = zzatVar2.zza;
                        int indexOf = str15.indexOf(str14);
                        if (indexOf >= 0) {
                            if (zzapVar instanceof zzai) {
                                zzapVar = ((zzai) zzapVar).zza(zzgVar, Arrays.asList(new zzat(str14), new zzah(Double.valueOf(indexOf)), zzatVar2));
                            }
                            zzahVar = new zzat(a.r(str15.substring(0, indexOf), zzapVar.zzi(), str15.substring(str14.length() + indexOf)));
                            return zzahVar;
                        }
                        return zzatVar2;
                    case 7:
                        zzh.zzj(FirebaseAnalytics.Event.SEARCH, 1, list);
                        if (Pattern.compile(list.isEmpty() ? "undefined" : zzgVar.zzb((zzap) list.get(0)).zzi()).matcher(this.zza).find()) {
                            return new zzah(Double.valueOf(matcher.start()));
                        }
                        return new zzah(Double.valueOf(-1.0d));
                    case '\b':
                        zzh.zzj("slice", 2, list);
                        String str16 = this.zza;
                        double zza2 = zzh.zza(!list.isEmpty() ? zzgVar.zzb((zzap) list.get(0)).zzh().doubleValue() : ShadowDrawableWrapper.COS_45);
                        if (zza2 < ShadowDrawableWrapper.COS_45) {
                            min = Math.max(str16.length() + zza2, (double) ShadowDrawableWrapper.COS_45);
                        } else {
                            min = Math.min(zza2, str16.length());
                        }
                        int i7 = (int) min;
                        double zza3 = zzh.zza(list.size() > 1 ? zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue() : str16.length());
                        if (zza3 < ShadowDrawableWrapper.COS_45) {
                            min2 = Math.max(str16.length() + zza3, (double) ShadowDrawableWrapper.COS_45);
                        } else {
                            min2 = Math.min(zza3, str16.length());
                        }
                        zzatVar3 = new zzat(str16.substring(i7, Math.max(0, ((int) min2) - i7) + i7));
                        return zzatVar3;
                    case '\t':
                        zzh.zzj("split", 2, list);
                        String str17 = this.zza;
                        if (str17.length() == 0) {
                            return new zzae(Arrays.asList(this));
                        }
                        ArrayList arrayList = new ArrayList();
                        if (list.isEmpty()) {
                            arrayList.add(this);
                        } else {
                            String zzi3 = zzgVar.zzb((zzap) list.get(0)).zzi();
                            long zzd = list.size() > 1 ? zzh.zzd(zzgVar.zzb((zzap) list.get(1)).zzh().doubleValue()) : 2147483647L;
                            if (zzd == 0) {
                                return new zzae();
                            }
                            String[] split = str17.split(Pattern.quote(zzi3), ((int) zzd) + 1);
                            int length2 = split.length;
                            if (zzi3.isEmpty() && length2 > 0) {
                                boolean isEmpty = split[0].isEmpty();
                                i3 = length2 - 1;
                                i4 = isEmpty;
                                i2 = isEmpty;
                                break;
                            } else {
                                i2 = 0;
                            }
                            i3 = length2;
                            i4 = i2;
                            if (length2 > zzd) {
                                i3--;
                            }
                            while (i4 < i3) {
                                arrayList.add(new zzat(split[i4]));
                                i4++;
                            }
                        }
                        return new zzae(arrayList);
                    case '\n':
                        zzh.zzj("substring", 2, list);
                        String str18 = this.zza;
                        if (list.isEmpty()) {
                            zzgVar2 = zzgVar;
                            i5 = 0;
                        } else {
                            zzgVar2 = zzgVar;
                            i5 = (int) zzh.zza(zzgVar2.zzb((zzap) list.get(0)).zzh().doubleValue());
                        }
                        if (list.size() > 1) {
                            length = (int) zzh.zza(zzgVar2.zzb((zzap) list.get(1)).zzh().doubleValue());
                        } else {
                            length = str18.length();
                        }
                        int min3 = Math.min(Math.max(i5, 0), str18.length());
                        int min4 = Math.min(Math.max(length, 0), str18.length());
                        zzatVar3 = new zzat(str18.substring(Math.min(min3, min4), Math.max(min3, min4)));
                        return zzatVar3;
                    case 11:
                        zzh.zzh(str2, 0, list);
                        return new zzat(this.zza.toUpperCase());
                    case '\f':
                        zzh.zzh("toLocaleLowerCase", 0, list);
                        return new zzat(this.zza.toLowerCase());
                    case '\r':
                        zzh.zzh("toLowerCase", 0, list);
                        return new zzat(this.zza.toLowerCase(Locale.ENGLISH));
                    case 14:
                        zzatVar2 = this;
                        zzh.zzh("toString", 0, list);
                        return zzatVar2;
                    case 15:
                        zzh.zzh("toUpperCase", 0, list);
                        return new zzat(this.zza.toUpperCase(Locale.ENGLISH));
                    case 16:
                        zzh.zzh("toUpperCase", 0, list);
                        return new zzat(this.zza.trim());
                    default:
                        throw new IllegalArgumentException("Command not supported");
                }
            }
        }
        str3 = "hasOwnProperty";
        switch (str.hashCode()) {
            case -1789698943:
                break;
            case -1776922004:
                break;
            case -1464939364:
                break;
            case -1361633751:
                break;
            case -1354795244:
                break;
            case -1137582698:
                break;
            case -906336856:
                break;
            case -726908483:
                break;
            case -467511597:
                break;
            case -399551817:
                break;
            case 3568674:
                break;
            case 103668165:
                break;
            case 109526418:
                break;
            case 109648666:
                break;
            case 530542161:
                break;
            case 1094496948:
                break;
            case 1943291465:
                break;
        }
        str6 = "undefined";
        switch (c2) {
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzap
    public final zzap zzd() {
        return new zzat(this.zza);
    }

    @Override // com.google.android.gms.internal.measurement.zzap
    public final Boolean zzg() {
        return Boolean.valueOf(!this.zza.isEmpty());
    }

    @Override // com.google.android.gms.internal.measurement.zzap
    public final Double zzh() {
        if (this.zza.isEmpty()) {
            return Double.valueOf((double) ShadowDrawableWrapper.COS_45);
        }
        try {
            return Double.valueOf(this.zza);
        } catch (NumberFormatException unused) {
            return Double.valueOf(Double.NaN);
        }
    }

    @Override // com.google.android.gms.internal.measurement.zzap
    public final String zzi() {
        return this.zza;
    }

    @Override // com.google.android.gms.internal.measurement.zzap
    public final Iterator zzl() {
        return new zzar(this);
    }
}
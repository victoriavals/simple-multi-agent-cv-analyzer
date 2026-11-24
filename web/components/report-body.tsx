"use client";

import * as Tabs from "@radix-ui/react-tabs";
import { domAnimation, LazyMotion, useReducedMotion } from "motion/react";
import * as m from "motion/react-m";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AnalysisReport } from "@/lib/analysis";

const SECTION_NAMES = {
  Ikhtisar: "overview", Overview: "overview",
  Kekuatan: "strengths", Strengths: "strengths",
  "Kesenjangan Keahlian": "gaps", "Skill Gaps": "gaps",
  "Rencana Upskilling": "plan", "Actionable Upskilling Plan": "plan",
  "Catatan Akhir": "notes", "Final Notes": "notes",
} as const;

type Section = (typeof SECTION_NAMES)[keyof typeof SECTION_NAMES];

function splitReport(markdown: string) {
  const headings = Array.from(markdown.matchAll(/^## ([^\r\n]+)[\t ]*\r?$/gm)).filter(
    (heading) => Object.hasOwn(SECTION_NAMES, heading[1].trim()),
  );
  const sections: Partial<Record<Section, string>> = {};
  headings.forEach((heading, index) => {
    const name = heading[1].trim() as keyof typeof SECTION_NAMES;
    sections[SECTION_NAMES[name]] = markdown.slice(heading.index + heading[0].length, headings[index + 1]?.index).trim();
  });
  return sections;
}

function MarkdownSection({ children, language }: { children: string; language: string }) {
  return (
    <div className="report-prose" lang={language}>
      <Markdown
        skipHtml
        disallowedElements={["img"]}
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}
      >
        {children}
      </Markdown>
    </div>
  );
}

export default function ReportBody({ report }: { report: AnalysisReport }) {
  const sections = splitReport(report.markdown);
  const language = report.language === "english" ? "en" : "id";
  const reducedMotion = useReducedMotion();

  return (
    <LazyMotion features={domAnimation}>
      <m.div initial={reducedMotion ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }}>
          <Tabs.Root defaultValue="summary" className="report-tabs">
            <Tabs.List aria-label="Bagian laporan" className="tab-list">
              <Tabs.Trigger value="summary">Ringkasan</Tabs.Trigger>
              <Tabs.Trigger value="skills">Skill</Tabs.Trigger>
              <Tabs.Trigger value="plan">Rencana belajar</Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="summary" className="tab-content">
              <MarkdownSection language={language}>{sections.overview || report.markdown}</MarkdownSection>
              {sections.notes && <div className="report-notes">
                <h3>Catatan dan sumber</h3>
                <MarkdownSection language={language}>{sections.notes}</MarkdownSection>
              </div>}
            </Tabs.Content>
            <Tabs.Content value="skills" className="tab-content">
              <h3>Kekuatan</h3>
              <MarkdownSection language={language}>{sections.strengths || "Data kekuatan tidak tersedia."}</MarkdownSection>
              <h3>Kesenjangan skill</h3>
              <MarkdownSection language={language}>{sections.gaps || "Data kesenjangan tidak tersedia."}</MarkdownSection>
            </Tabs.Content>
            <Tabs.Content value="plan" className="tab-content">
              <MarkdownSection language={language}>{sections.plan || "Rencana belajar tidak tersedia."}</MarkdownSection>
            </Tabs.Content>
          </Tabs.Root>
      </m.div>
    </LazyMotion>
  );
}
